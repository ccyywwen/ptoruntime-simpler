# Host Map 驱动实验说明

这份说明用于把 `w0-host-map-refactor` 分支里的 L3-L2 host-map 实验复现。所有交付入口都在 `tools/host_map_test/` 下。

## 文件入口

| 文件 | 作用 |
| --- | --- |
| `tools/host_map_test/host_map_test.py` | host-register 最小实验入口。覆盖两类 L3 侧自读写验证，以及 host-register 后由 AICPU observer 读取的复现入口。 |
| `tools/host_map_test/l3_l2_device_observer.cpp` | `host_map_test.py` 编译并提交到 AICPU 的 observer kernel。它等待 tail counter、读取 payload、再写 completion counter。 |
| `tools/host_map_test/run_acl_copy_stream_test.py` | 现有 VMM+ACL-COPY 生产链路的包装入口。它强制 L3-L2 region 使用 `ONBOARD_VMM`，复用 `examples/workers/l3/l3_l2_orch_comm_stream` 的闭环 stream 场景。 |


## 环境准备

从仓库根目录执行，使用当前分支本地 venv：

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --no-build-isolation -e .
```

硬件运行前先确认平台匹配：

```bash
.claude/skills/onboard-arch-precheck/check.sh a2a3
```

我们本机检测结果是 `a2a3|Ascend910_9392|Ascend910_93`。

建议每次实验单独建输出目录，并把 device log 放进去：

```bash
RUN_DIR="outputs/<case>_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/ascend"
. .venv/bin/activate
export ASCEND_PROCESS_LOG_PATH="$PWD/$RUN_DIR/ascend"
```

## 实验一：L3 侧 direct `halHostRegister` 自读写

命令：

```bash
python tools/host_map_test/host_map_test.py   --platform a2a3   --device 14   --case l3-direct-rw   --output "$RUN_DIR/result.json"
```

链路：

1. L2 child 侧用 direct device allocation 创建 region。
2. L3 侧对 device VA 调 `halHostRegister(... DEV_SVM_MAP_HOST ...)`。
3. L3 侧通过返回的 host VA 写 8-byte payload，再从同一 host VA 读回。
4. L3 侧通过同一 mapping 写 tail counter，再读回 counter。

我们的结果：通过。`payload_u64=8097873951909177424`，`counter=1`。

## 实验二：L3 侧 VMM import 后 `halHostRegister` 自读写

命令：

```bash
python tools/host_map_test/host_map_test.py   --platform a2a3   --device 14   --case l3-vmm-host-register-rw   --output "$RUN_DIR/result.json"
```

链路：

1. L2 child 侧创建 VMM device memory：`aclrtMallocPhysical`、`aclrtReserveMemAddress`、`aclrtMapMem`、`aclrtMemSetAccess`。
2. L2 child 侧导出 shareable handle：`aclrtMemExportToShareableHandle`。
3. L3 侧 import shareable handle：`aclrtMemImportFromShareableHandle`，再 reserve/map/set-access 得到 L3 侧 VMM VA。
4. L3 侧对这个 imported VMM VA 调 `halHostRegister(... DEV_SVM_MAP_HOST ...)`。
5. L3 侧通过返回的 host VA 写/读 payload，并写/读 counter。

我们的结果：通过。`payload_u64=8097873951909177424`，`counter=1`。


## 实验三：现有 VMM+ACL-COPY 闭环 stream

命令：

```bash
python tools/host_map_test/run_acl_copy_stream_test.py   --platform a2a3   --device 14   --output "$RUN_DIR/result.json"
```

链路：

1. 包装脚本强制 `Worker._select_l3_l2_region_backend` 返回 `ONBOARD_VMM` / backend `none`。
2. L2 child 侧创建并导出 VMM region；L3 侧 import 后不调用 `halHostRegister`。
3. L3 侧 `region.payload_write` / counter notify 走 ACL H2D copy。
4. AICPU orchestration 通过现有 L3-L2 endpoint 读取 header/input 和 counter，调度 AIV 处理。
5. AICPU 写 completion/output，L3 侧通过 ACL D2H copy 读回并校验输出。

我们的结果：通过。说明现有 VMM+ACL-COPY 方式可以正确读写，且 AICPU 能正常读到 L3 发布的数据。


## 实验四：VMM + SharedHandle + `halHostRegister` 后 AICPU observer 读取

命令：

```bash
python tools/host_map_test/host_map_test.py   --platform a2a3   --device 14   --case vmm-host-register-observer   --output "$RUN_DIR/result.json"
```

链路和实验二相同，区别是 L3 写 payload/tail 后，由 AICPU observer 直接读取 tail 和 payload，再写 completion。

我们的结果：失败复现。`primitive_result.status=supported`，但 `case_results.vmm-host-register-observer.status=probe_error`，reason 为 `SIGNAL_WAIT timed out; observed=0`。host 侧看到 `507018`，runtime 分类为 `orch_error_code=9 EXPLICIT_ORCH_FATAL`。


## 实验五：direct `halHostRegister` 后 AICPU observer 读取

命令：

```bash
python tools/host_map_test/host_map_test.py   --platform a2a3   --device 14   --case direct-host-register-observer   --output "$RUN_DIR/result.json"
```

链路和实验一相同，区别是 L3 写 payload/tail 后，由 AICPU observer 读取。

我们的结果：失败复现。`primitive_result.status=supported`，但 `case_results.direct-host-register-observer.status=probe_error`，reason 为 `SIGNAL_WAIT timed out; observed=0`。host 侧看到 `507018`，runtime 分类为 `orch_error_code=9 EXPLICIT_ORCH_FATAL`。


## 对照结论

| 实验 | L3 自己读写 | AICPU 能否读到 | 本机结果 |
| --- | --- | --- | --- |
| direct `halHostRegister` | 是 | 否 | L3 自读写通过，AICPU observer 超时看到 `observed=0` |
| VMM + SharedHandle + `halHostRegister` | 是 | 否 | L3 自读写通过，AICPU observer 超时看到 `observed=0` |
| VMM + ACL-COPY | 是 | 是 | 闭环 stream 通过，输出校验通过 |

因此当前现象不是 `halHostRegister` primitive 调用失败，也不是 L3 host VA 不可读写；问题集中在 host-registered mapping 经 L3 写入后，AICPU 侧是否能观察到这些写入。

## 我们本机环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-07-31 |
| 分支 | `w0-host-map-refactor` |
| 平台 | `a2a3` |
| SoC | `Ascend910_9392`，`Short_SoC_version=Ascend910_93` |
| Python | `3.9.9` |
| Runtime | `tensormap_and_ringbuffer` |
| Driver package | `25.5.1` (`/usr/local/Ascend/driver/version.info`) |
| `npu-smi` | `25.5.1` |
| `ascendhal_version` | `7.35.23` |
| Driver innerversion | `V100R001C23SPC006B220` |
| CANN toolkit | `Ascend-cann-toolkit 9.1.0`, innerversion `V100R001C25B114`, path `/usr/local/Ascend/cann-9.1.0` |
| CANN ops | `Ascend-cann-A3-ops 9.1.0`, innerversion `V100R001C25B114` |
| OPP | `9.1.0`，timestamp `20260602_130307579` |
| CANN library path | `LD_LIBRARY_PATH` 指向 `/usr/local/Ascend/cann-9.1.0/lib64`、`/usr/local/Ascend/cann-9.1.0/opp/...`、`/usr/local/Ascend/driver/lib64` |
| CANN Python path | `PYTHONPATH` 包含 `/usr/local/Ascend/cann-9.1.0/python/site-packages` 和 `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe` |
| 设备 | card 7 physical device `14` |
| 隔离方式 | 本机 `task-submit` 不在 PATH，实验为 unlocked run |
| 安装方式 | `.venv`，`pip install --no-build-isolation -e .` |
