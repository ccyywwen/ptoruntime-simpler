#!/usr/bin/env python3
"""Run the existing L3-L2 stream demo with the VMM ACL-copy backend forced."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from examples.workers.l3.l3_l2_orch_comm_stream.test_l3_l2_orch_comm_stream import run_closed_loop_stream
from simpler import worker as worker_module
from simpler.l3_l2_orch_comm import L3L2RegionAccessProfile


def _force_acl_copy(self, worker_id: int):
    return worker_module._RegionDataPlaneDecision(
        L3L2RegionAccessProfile.ONBOARD_VMM,
        worker_module._RegionHostMapBackend.NONE,
        "forced",
        "host_map_test_acl_stream",
        "forced VMM ACL-copy baseline",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("a2a3", "a5"))
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    worker_module.Worker._select_l3_l2_region_backend = _force_acl_copy
    artifact = {
        "case": "vmm-acl-copy-stream",
        "platform": args.platform,
        "device_id": int(args.device),
        "runtime": "tensormap_and_ringbuffer",
        "timestamp_s": int(time.time()),
        "access_profile": "ONBOARD_VMM",
        "backend": "none",
    }
    try:
        run_closed_loop_stream(args.platform, int(args.device))
    except Exception as exc:  # noqa: BLE001
        artifact.update({"status": "probe_error", "reason": str(exc)})
    else:
        artifact.update(
            {
                "status": "pass",
                "reason": "VMM ACL-copy closed-loop stream passed; L3 wrote input/header/counters, AICPU/AIV consumed them, and L3 read verified output.",
            }
        )

    text = json.dumps(artifact, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
