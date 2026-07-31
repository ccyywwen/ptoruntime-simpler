#!/usr/bin/env python3
"""Host-map visibility tests for L3-L2 region backends."""

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
    _memory_wmb_for_test,
)
from simpler.l3_l2_orch_comm import L3L2RegionAccessProfile, NotifyOp, WaitCmp
from simpler.task_interface import CallConfig, ChipCallable, TaskArgs
from simpler.worker import Worker, _RegionDataPlaneDecision, _RegionHostMapBackend
from simpler_setup.kernel_compiler import KernelCompiler


_PAYLOAD_BYTES = 64
_COUNTER_BYTES = 128
_REGION_TAIL_COUNTER = 0
_REGION_COMPLETION_COUNTER = 4
_EXPECTED_FIRST_U64 = 0x70616D74736F6850
_EXPECTED_TAIL = 1
_OBSERVER_SOURCE = Path(__file__).with_name("l3_l2_device_observer.cpp")

_CASES = (
    "l3-direct-rw",
    "l3-vmm-host-register-rw",
    "vmm-host-register-observer",
    "direct-host-register-observer",
    "all",
)


def _result(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"status": status, "reason": reason}
    data.update(extra)
    return data


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
        func_name="host_map_device_observer",
        binary=orch,
        children=[],
    )


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


def _force_decision(access_profile: L3L2RegionAccessProfile, backend: _RegionHostMapBackend, reason: str):
    def select(_worker_id: int) -> _RegionDataPlaneDecision:
        return _RegionDataPlaneDecision(access_profile, backend, "forced", "host_map_test", reason)

    return select


def _new_worker(platform: str, device: int, runtime: str) -> Worker:
    worker = Worker(level=3, device_ids=[int(device)], platform=platform, runtime=runtime)
    worker.init()
    return worker


def _l3_readwrite_result(platform: str, device: int, runtime: str, backend: _RegionHostMapBackend) -> dict[str, Any]:
    worker = _new_worker(platform, device, runtime)
    worker._select_l3_l2_region_backend = _force_decision(  # type: ignore[method-assign]
        L3L2RegionAccessProfile.ONBOARD_HOST_REGISTERED,
        backend,
        f"force {backend.value} L3 self read/write",
    )
    try:
        config = CallConfig()
        config.aicpu_thread_num = 2
        observed: dict[str, Any] = {}

        def orch(orch_handle, _args, _cfg):
            region = orch_handle.create_l3_l2_region(
                worker_id=0, payload_bytes=_PAYLOAD_BYTES, counter_bytes=_COUNTER_BYTES
            )
            try:
                payload = ctypes.create_string_buffer(_PAYLOAD_BYTES)
                struct.pack_into("<Q", payload, 0, _EXPECTED_FIRST_U64)
                region.payload_write(0, payload, nbytes=8)
                _memory_wmb_for_test()
                readback = ctypes.create_string_buffer(8)
                region.payload_read(0, readback, nbytes=8)
                got_payload = struct.unpack_from("<Q", readback.raw, 0)[0]
                if got_payload != _EXPECTED_FIRST_U64:
                    raise RuntimeError(f"payload readback mismatch expected={_EXPECTED_FIRST_U64} got={got_payload}")
                counter = region.counter(_REGION_TAIL_COUNTER)
                counter.notify(_EXPECTED_TAIL, NotifyOp.Set)
                counter_result = counter.test(_EXPECTED_TAIL, WaitCmp.EQ)
                if not counter_result.matched:
                    raise RuntimeError(f"counter readback mismatch observed={counter_result.observed}")
                observed["payload_u64"] = got_payload
                observed["counter"] = int(counter_result.observed)
            finally:
                region.free()

        worker.run(orch, args=None, config=config)
        return _result(
            "pass",
            "L3 host write/read and counter self-check passed through host-registered mapping",
            backend=backend.value,
            access_profile=L3L2RegionAccessProfile.ONBOARD_HOST_REGISTERED.name,
            validation="l3_payload_counter_readwrite",
            observed=observed,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "probe_error",
            str(exc),
            backend=backend.value,
            access_profile=L3L2RegionAccessProfile.ONBOARD_HOST_REGISTERED.name,
            validation="l3_payload_counter_readwrite",
        )
    finally:
        worker.close()


def _observer_result(
    platform: str,
    device: int,
    runtime: str,
    access_profile: L3L2RegionAccessProfile,
    backend: _RegionHostMapBackend,
    observer: ChipCallable,
) -> dict[str, Any]:
    worker = Worker(level=3, device_ids=[int(device)], platform=platform, runtime=runtime)
    observer_handle = worker.register(observer)
    worker.init()
    worker._select_l3_l2_region_backend = _force_decision(  # type: ignore[method-assign]
        access_profile,
        backend,
        f"force {access_profile.name} {backend.value} observer test",
    )
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
        return _result(
            "pass",
            "AICPU observer saw payload and tail, then published completion",
            backend=backend.value,
            access_profile=access_profile.name,
            validation="aicpu_payload_tail_visibility",
        )
    except Exception as exc:  # noqa: BLE001
        if worker._is_known_host_map_unsupported(exc):
            status = "unsupported"
        else:
            status = "probe_error"
        return _result(
            status,
            str(exc),
            backend=backend.value,
            access_profile=access_profile.name,
            validation="aicpu_payload_tail_visibility",
        )
    finally:
        worker.close()


def _run_case(case: str, platform: str, device: int, runtime: str, observer: ChipCallable) -> dict[str, Any]:
    if case == "l3-direct-rw":
        return _l3_readwrite_result(platform, device, runtime, _RegionHostMapBackend.DIRECT_HAL_HOST_REGISTER)
    if case == "l3-vmm-host-register-rw":
        return _l3_readwrite_result(platform, device, runtime, _RegionHostMapBackend.VMM_IMPORT_THEN_HOST_REGISTER)
    if case == "vmm-host-register-observer":
        return _observer_result(
            platform,
            device,
            runtime,
            L3L2RegionAccessProfile.ONBOARD_HOST_REGISTERED,
            _RegionHostMapBackend.VMM_IMPORT_THEN_HOST_REGISTER,
            observer,
        )
    if case == "direct-host-register-observer":
        return _observer_result(
            platform,
            device,
            runtime,
            L3L2RegionAccessProfile.ONBOARD_HOST_REGISTERED,
            _RegionHostMapBackend.DIRECT_HAL_HOST_REGISTER,
            observer,
        )
    raise ValueError(f"unknown case {case}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("a2a3", "a5"))
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--output", default=None)
    parser.add_argument("--case", choices=_CASES, default="vmm-host-register-observer")
    args = parser.parse_args()

    observer = _build_observer_callable(args.platform, args.runtime)
    selected_cases = [c for c in _CASES if c != "all"] if args.case == "all" else [args.case]
    case_results = {case: _run_case(case, args.platform, int(args.device), args.runtime, observer) for case in selected_cases}
    primitive = dict(_host_map_capability_probe(int(args.device)))

    artifact = {
        "platform": args.platform,
        "device_id": int(args.device),
        "runtime": args.runtime,
        "timestamp_s": int(time.time()),
        "primitive_result": primitive,
        "case_results": case_results,
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
