#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import ctypes
import os
import time
from pathlib import Path
from typing import Any

from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, ChipStorageTaskArgs, CoreCallable, DataType, Tensor
from simpler.worker import Worker
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root


_KERNEL_DIR = Path(__file__).with_name("dfx_host_map_refill_validation_kernels")
_ORCH_SOURCE = _KERNEL_DIR / "high_task_refill_orch.cpp"
_NOOP_AIC_SOURCE = _KERNEL_DIR / "noop_aic.cpp"
_TASK_RECORDS_PER_BUFFER = 1000
_ORCH_RECORDS_PER_BUFFER = 16384
_FREE_QUEUE_SLOTS = 4
_TASK_INITIAL_RECORD_CAPACITY = _TASK_RECORDS_PER_BUFFER * _FREE_QUEUE_SLOTS
_ORCH_INITIAL_RECORD_CAPACITY = _ORCH_RECORDS_PER_BUFFER * _FREE_QUEUE_SLOTS


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _build_callable(platform: str, runtime: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root(verbose=True)
    build_dir = Path("build") / "dfx_host_map_refill_validation"
    build_dir.mkdir(parents=True, exist_ok=True)
    incore = compiler.compile_incore(
        str(_NOOP_AIC_SOURCE), core_type="aic", pto_isa_root=pto_isa_root, build_dir=str(build_dir)
    )
    if not platform.endswith("sim"):
        incore = extract_text_section(incore)
    orch = compiler.compile_orchestration(runtime, str(_ORCH_SOURCE), build_dir=str(build_dir))
    child = CoreCallable.build([ArgDirection.IN], incore)
    return ChipCallable.build(
        signature=[ArgDirection.IN],
        func_name="dfx_host_map_refill_high_task_orch",
        binary=orch,
        children=[(0, child)],
    )


def _count_nested(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    total = 0
    for item in values:
        if isinstance(item, list) and item and all(not isinstance(v, list) for v in item):
            total += 1
        elif isinstance(item, list):
            total += _count_nested(item)
    return total


def _max_first_field_count(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    counts: dict[int, int] = {}
    for record in records:
        if isinstance(record, list) and record:
            key = int(record[0])
            counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def _read_log_markers(log_root: Path | None) -> dict[str, int]:
    markers = {
        "free_queue_empty_triggered": 0,
        "ready_queue_full_triggered": 0,
        "backpressure_released": 0,
        "free_queue_publish_failed": 0,
    }
    if log_root is None or not log_root.exists():
        return markers
    for path in log_root.glob("device-*/device-*.log"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        markers["free_queue_empty_triggered"] += text.count("free-queue-empty")
        markers["ready_queue_full_triggered"] += text.count("ready-queue-full")
        markers["backpressure_released"] += text.count("DFX backpressure RELEASED")
        markers["free_queue_publish_failed"] += text.count("failed to publish free_queue")
    return markers


def _run(platform: str, device: int, runtime: str, task_count: int, output_prefix: Path) -> dict[str, Any]:
    callable_ = _build_callable(platform, runtime)
    worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
    handle = worker.register(callable_)
    worker.init()
    try:
        input_storage = (ctypes.c_float * 1)(0.0)
        args = ChipStorageTaskArgs()
        args.add_tensor(Tensor.make(ctypes.addressof(input_storage), (1,), DataType.FLOAT32))
        args.add_scalar(task_count)
        config = CallConfig()
        config.aicpu_thread_num = 2
        config.enable_l2_swimlane = 4
        config.runtime_env.ring_task_window = 131072
        config.runtime_env.ring_heap = 131072
        config.runtime_env.ring_dep_pool = 131072
        config.output_prefix = str(output_prefix)
        worker.run(handle, args, config)
    finally:
        worker.close()

    swimlane_path = output_prefix / "l2_swimlane_records.json"
    with swimlane_path.open() as f:
        swimlane = json.load(f)

    orch_phase_records = _count_nested(swimlane.get("aicpu_orchestrator_phases", []))
    sched_phase_records = _count_nested(swimlane.get("aicpu_scheduler_phases", []))
    aicpu_task_records = swimlane.get("aicpu_tasks", [])
    aicore_task_records = swimlane.get("aicore_tasks", [])
    aicpu_tasks = _count_nested(aicpu_task_records)
    aicore_tasks = _count_nested(aicore_task_records)
    max_aicpu_task_records_per_core = _max_first_field_count(aicpu_task_records)
    max_aicore_task_records_per_core = _max_first_field_count(aicore_task_records)
    exceeded_task_initial_capacity = (
        max_aicpu_task_records_per_core > _TASK_INITIAL_RECORD_CAPACITY
        or max_aicore_task_records_per_core > _TASK_INITIAL_RECORD_CAPACITY
    )
    log_root_env = os.environ.get("ASCEND_PROCESS_LOG_PATH")
    log_root = Path(log_root_env) if log_root_env else None

    return {
        "status": "pass" if exceeded_task_initial_capacity else "inconclusive",
        "reason": "per-core task records exceeded initial free_queue capacity"
        if exceeded_task_initial_capacity
        else "per-core task records did not exceed initial free_queue capacity",
        "platform": platform,
        "device": device,
        "runtime": runtime,
        "task_count": task_count,
        "l2_swimlane_level": swimlane.get("l2_swimlane_level"),
        "orch_phase_records": orch_phase_records,
        "sched_phase_records": sched_phase_records,
        "aicpu_task_records": aicpu_tasks,
        "aicore_task_records": aicore_tasks,
        "task_initial_record_capacity": _TASK_INITIAL_RECORD_CAPACITY,
        "max_aicpu_task_records_per_core": max_aicpu_task_records_per_core,
        "max_aicore_task_records_per_core": max_aicore_task_records_per_core,
        "exceeded_task_initial_capacity": exceeded_task_initial_capacity,
        "orch_initial_record_capacity": _ORCH_INITIAL_RECORD_CAPACITY,
        "exceeded_orch_initial_capacity": orch_phase_records > _ORCH_INITIAL_RECORD_CAPACITY,
        "log_markers": _read_log_markers(log_root),
        "swimlane_artifact": _rel(swimlane_path),
        "timestamp_s": int(time.time()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DFX runtime host free_queue refill visibility.")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--task-count", type=int, default=70000)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    output_prefix.mkdir(parents=True, exist_ok=True)
    artifact = _run(args.platform, args.device, args.runtime, args.task_count, output_prefix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
