#!/usr/bin/env python3
"""Developer validation for L3-L2 host-map backend candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _task_interface import _host_map_capability_probe  # pyright: ignore[reportMissingImports]
from simpler.worker import Worker, _RegionHostMapBackend


def _result(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"status": status, "reason": reason}
    data.update(extra)
    return data


def _candidate_result(worker: Worker, backend: _RegionHostMapBackend) -> dict[str, Any]:
    try:
        reason = worker._probe_selected_l3_l2_host_map_backend(0, backend)
    except Exception as exc:  # noqa: BLE001
        if worker._is_known_host_map_unsupported(exc):
            return _result("unsupported", str(exc), backend=backend.value)
        return _result("probe_error", str(exc), backend=backend.value)
    return _result("pass", reason, backend=backend.value)


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
    args = parser.parse_args()

    worker = Worker(level=3, device_ids=[int(args.device)], platform=args.platform, runtime=args.runtime)
    worker.init()
    try:
        primitive = dict(_host_map_capability_probe(int(args.device)))
        direct = _candidate_result(worker, _RegionHostMapBackend.DIRECT_HAL_HOST_REGISTER)
        vmm = _candidate_result(worker, _RegionHostMapBackend.VMM_IMPORT_THEN_HOST_REGISTER)
    finally:
        worker.close()

    artifact = {
        "HostMapBackend": _select_backend(direct, vmm),
        "platform": args.platform,
        "device_id": int(args.device),
        "runtime": args.runtime,
        "timestamp_s": int(time.time()),
        "primitive_result": primitive,
        "direct_result": direct,
        "vmm_import_result": vmm,
        "reason": {
            "winner_rule": "DIRECT pass, else VMM-import pass, else NONE",
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
