#!/usr/bin/env python3
"""Developer validation for L3-L2 host-map backend candidates."""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
import time
from pathlib import Path
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _host_map_capability_probe,
    _l3_host_mapped_counter_notify,
    _l3_host_mapped_counter_test,
    _l3_host_mapped_counter_wait,
    _l3_host_mapped_payload_write,
    _l3_host_mapped_region_close,
    _l3_host_mapped_region_register_onboard_direct,
    _memory_wmb_for_test,
)
from simpler.l3_l2_orch_comm import L3L2RegionAccessProfile, NotifyOp, WaitCmp
from simpler.task_interface import CallConfig, ChipCallable, TaskArgs
from simpler.worker import Worker, _RegionDataPlaneDecision, _RegionHostMapBackend
from simpler_setup.kernel_compiler import KernelCompiler


def _result(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"status": status, "reason": reason}
    data.update(extra)
    return data


_PAYLOAD_BYTES = 64
_COUNTER_BYTES = 128
_REGION_TAIL_COUNTER = 0
_REGION_COMPLETION_COUNTER = 4
_RAW_TAIL_OFFSET = _PAYLOAD_BYTES
_RAW_COMPLETION_OFFSET = _PAYLOAD_BYTES + 4
_RAW_TOTAL_BYTES = _PAYLOAD_BYTES + _COUNTER_BYTES
_EXPECTED_FIRST_U64 = 0x70616D74736F6850
_EXPECTED_TAIL = 1
_OBSERVER_SOURCE = Path(__file__).with_name("phase0_host_map_validation_kernels") / "l3_l2_device_observer.cpp"


def _build_observer_callable(platform: str, runtime: str) -> ChipCallable:
    kc = KernelCompiler(platform=platform)
    extra_common = [str(kc.project_root / "src" / "common")]
    orch = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=str(_OBSERVER_SOURCE),
        extra_include_dirs=extra_common,
    )
    return ChipCallable.build(
        signature=[],
        func_name="phase0_host_map_device_observer",
        binary=orch,
        children=[],
    )


def _observer_args_raw(device_addr: int) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(0)
    args.add_scalar(int(device_addr))
    for _ in range(5):
        args.add_scalar(0)
    args.add_scalar(0)
    args.add_scalar(_EXPECTED_FIRST_U64)
    args.add_scalar(_RAW_TAIL_OFFSET)
    args.add_scalar(_RAW_COMPLETION_OFFSET)
    args.add_scalar(_EXPECTED_TAIL)
    return args


def _observer_args_region(region) -> TaskArgs:
    args = TaskArgs()
    args.add_scalar(1)
    for scalar in region.descriptor_scalars():
        args.add_scalar(int(scalar))
    args.add_scalar(0)
    args.add_scalar(_EXPECTED_FIRST_U64)
    args.add_scalar(_REGION_TAIL_COUNTER)
    args.add_scalar(_REGION_COMPLETION_COUNTER)
    args.add_scalar(_EXPECTED_TAIL)
    return args


def _force_backend(backend: _RegionHostMapBackend):
    def select(_worker_id: int) -> _RegionDataPlaneDecision:
        return _RegionDataPlaneDecision(
            L3L2RegionAccessProfile.ONBOARD_HOST_REGISTERED,
            backend,
            "supported",
            "phase0_device_observer",
            "forced Phase 0 candidate",
        )

    return select


def _publish_payload_and_tail(handle: int, payload_offset: int, tail_offset: int) -> None:
    payload = ctypes.create_string_buffer(_PAYLOAD_BYTES)
    struct.pack_into("<Q", payload, 0, _EXPECTED_FIRST_U64)
    _l3_host_mapped_payload_write(int(handle), int(payload_offset), ctypes.addressof(payload), 8)
    _memory_wmb_for_test()
    _l3_host_mapped_counter_notify(int(handle), int(tail_offset), _EXPECTED_TAIL, int(NotifyOp.Set))


def _local_allocation_result(platform: str, device: int, runtime: str, observer: ChipCallable) -> dict[str, Any]:
    worker = Worker(level=2, device_id=int(device), platform=platform, runtime=runtime)
    observer_handle = worker.register(observer)
    worker.init()
    dev_addr = 0
    handle = 0
    try:
        config = CallConfig()
        config.aicpu_thread_num = 2
        dev_addr = int(worker.malloc(_RAW_TOTAL_BYTES))
        handle = int(_l3_host_mapped_region_register_onboard_direct(int(device), dev_addr, _RAW_TOTAL_BYTES))
        _publish_payload_and_tail(handle, 0, _RAW_TAIL_OFFSET)
        worker.run(observer_handle, _observer_args_raw(dev_addr), config)
        status, error_kind, observed, matched, message = _l3_host_mapped_counter_wait(
            handle, _RAW_COMPLETION_OFFSET, _EXPECTED_TAIL, int(WaitCmp.GE), 5_000_000_000
        )
        if status != 0 or not matched:
            return _result(
                "probe_error",
                f"completion counter not visible status={status} error_kind={error_kind} observed={observed} message={message}",
                validation="local_allocation_dfx_slot_tail",
                device_addr=dev_addr,
            )
        completion_matched, completion_observed = _l3_host_mapped_counter_test(
            handle, _RAW_COMPLETION_OFFSET, _EXPECTED_TAIL, int(WaitCmp.GE)
        )
        return _result(
            "pass",
            "AICPU observer saw local host-registered payload and tail writes",
            validation="local_allocation_dfx_slot_tail",
            device_addr=dev_addr,
            completion_observed=completion_observed,
            completion_matched=completion_matched,
        )
    except Exception as exc:  # noqa: BLE001
        return _result("probe_error", str(exc), validation="local_allocation_dfx_slot_tail", device_addr=dev_addr)
    finally:
        if handle:
            try:
                _l3_host_mapped_region_close(handle)
            except Exception:
                pass
        if dev_addr:
            try:
                worker.free(dev_addr)
            except Exception:
                pass
        worker.close()


def _candidate_result(platform: str, device: int, runtime: str, observer: ChipCallable, backend: _RegionHostMapBackend) -> dict[str, Any]:
    worker = Worker(level=3, device_ids=[int(device)], platform=platform, runtime=runtime)
    observer_handle = worker.register(observer)
    worker.init()
    worker._select_l3_l2_region_backend = _force_backend(backend)  # type: ignore[method-assign]
    try:
        config = CallConfig()
        config.aicpu_thread_num = 2

        def orch(orch_handle, _args, cfg):
            region = orch_handle.create_l3_l2_region(
                worker_id=0, payload_bytes=_PAYLOAD_BYTES, counter_bytes=_COUNTER_BYTES
            )
            try:
                completion = region.counter(_REGION_COMPLETION_COUNTER)
                data_ready = region.counter(_REGION_TAIL_COUNTER)
                orch_handle.submit_next_level(observer_handle, _observer_args_region(region), cfg, worker=0)
                payload = ctypes.create_string_buffer(_PAYLOAD_BYTES)
                struct.pack_into("<Q", payload, 0, _EXPECTED_FIRST_U64)
                region.payload_write(0, payload, nbytes=8)
                _memory_wmb_for_test()
                data_ready.notify(_EXPECTED_TAIL, NotifyOp.Set)
                completion.wait(_EXPECTED_TAIL, WaitCmp.GE, timeout=5.0)
            finally:
                region.free()

        worker.run(orch, args=None, config=config)
    except Exception as exc:  # noqa: BLE001
        if worker._is_known_host_map_unsupported(exc):
            return _result("unsupported", str(exc), backend=backend.value, validation="region_dfx_slot_tail")
        return _result("probe_error", str(exc), backend=backend.value, validation="region_dfx_slot_tail")
    finally:
        worker.close()
    return _result(
        "pass",
        "AICPU observer saw L3-L2 region host-registered payload and tail writes",
        backend=backend.value,
        validation="region_dfx_slot_tail",
    )


def _select_backend(direct: dict[str, Any], vmm: dict[str, Any]) -> str:
    if direct.get("status") == "pass":
        return "DIRECT_HAL_HOST_REGISTER"
    if vmm.get("status") == "pass":
        return "VMM_IMPORT_THEN_HOST_REGISTER"
    return "NONE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("a2a3", "a5"))
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--output", default=None)
    parser.add_argument("--candidate", choices=("both", "direct", "vmm"), default="both")
    parser.add_argument("--experiment", choices=("all", "local", "region"), default="all")
    args = parser.parse_args()

    observer = _build_observer_callable(args.platform, args.runtime)
    local = _result("skipped", "experiment not requested", validation="local_allocation_dfx_slot_tail")
    direct = _result("skipped", "candidate not requested", backend=_RegionHostMapBackend.DIRECT_HAL_HOST_REGISTER.value)
    vmm = _result("skipped", "candidate not requested", backend=_RegionHostMapBackend.VMM_IMPORT_THEN_HOST_REGISTER.value)
    if args.experiment in ("all", "local"):
        local = _local_allocation_result(args.platform, int(args.device), args.runtime, observer)
    if args.experiment in ("all", "region"):
        if args.candidate in ("both", "vmm"):
            vmm = _candidate_result(
                args.platform, int(args.device), args.runtime, observer, _RegionHostMapBackend.VMM_IMPORT_THEN_HOST_REGISTER
            )
        if args.candidate in ("both", "direct"):
            direct = _candidate_result(
                args.platform, int(args.device), args.runtime, observer, _RegionHostMapBackend.DIRECT_HAL_HOST_REGISTER
            )
    primitive = dict(_host_map_capability_probe(int(args.device)))

    artifact = {
        "HostMapBackend": _select_backend(direct, vmm),
        "platform": args.platform,
        "device_id": int(args.device),
        "runtime": args.runtime,
        "timestamp_s": int(time.time()),
        "primitive_result": primitive,
        "local_allocation_result": local,
        "direct_result": direct,
        "vmm_import_result": vmm,
        "reason": {
            "winner_rule": "DIRECT pass, else VMM-import pass, else NONE",
            "local_status": local.get("status"),
            "direct_status": direct.get("status"),
            "vmm_import_status": vmm.get("status"),
        },
    }

    text = json.dumps(artifact, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
