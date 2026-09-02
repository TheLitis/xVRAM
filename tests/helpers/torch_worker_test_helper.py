from __future__ import annotations

import os
import sys
import time

from xvram.torch_protocol import MessageType, read_frame, write_frame
from xvram.torch_report import ALLOWLIST_HASH, EXIT_COMPLETED, EXIT_RUNTIME, empty_report


def completed_report(plan: dict[str, object]) -> dict[str, object]:
    report = empty_report(plan, exit_code=EXIT_COMPLETED, status="completed", message="helper")
    report["device"]["total_vram_bytes"] = 8 * 1024**3
    report["model"].update(
        {"logical_bytes": 1152, "parameter_bytes": 1024, "activation_bytes": 128}
    )
    report["graph"].update(
        {
            "hash": "a" * 64,
            "backend_hash": "a" * 64,
            "allowlist_hash": ALLOWLIST_HASH,
            "node_count": 1,
            "region_count": 1,
        }
    )
    report["execution"].update(
        {
            "leases_acquired": 1,
            "leases_sealed": 1,
            "leases_retired": 1,
            "events_recorded": 1,
            "events_retired": 1,
            "live_views_peak": 1,
            "regions_completed": 1,
            "region_timings_ms": [0.1],
            "trace_records": 1,
        }
    )
    report["proof"]["stable_addresses"] = True
    report["cache"].update(
        {"maps": 1, "set_access": 1, "unmaps": 1, "misses": 1, "h2d_bytes": 128}
    )
    report["verification"].update(
        {"reference_digest": "a" * 64, "output_digest": "a" * 64}
    )
    for key in report["cleanup"]:
        if key not in {"worker_reaped", "complete"}:
            report["cleanup"][key] = True
    return report


def main() -> int:
    mode = sys.argv[1]
    plan = read_frame(sys.stdin.buffer).payload
    if mode == "crash":
        return 19
    if mode == "hang":
        time.sleep(30)
        return 0
    if mode == "heartbeat-hang":
        sequence = 0
        while True:
            sequence += 1
            write_frame(
                sys.stdout.buffer,
                MessageType.HEARTBEAT,
                {"sequence": sequence, "monotonic_ns": time.monotonic_ns()},
            )
            time.sleep(0.05)
    if mode == "truncated":
        sys.stdout.buffer.write(b"XVT1\x01\x05\x00\x00\x00\x10{}")
        sys.stdout.buffer.flush()
        return 0
    if mode == "oversized":
        sys.stdout.buffer.write(b"XVT1\x01\x05\x00\x10\x00\x01")
        sys.stdout.buffer.flush()
        return 0
    report = completed_report(plan)
    write_frame(
        sys.stdout.buffer,
        MessageType.PROGRESS,
        {"sequence": 1, "operations_retired": 1},
    )
    if mode == "malformed-final":
        report["execution"]["leases_acquired"] = "not-an-integer"
    if mode == "nonmonotonic-protocol":
        write_frame(
            sys.stdout.buffer,
            MessageType.HEARTBEAT,
            {"sequence": 1, "monotonic_ns": time.monotonic_ns()},
        )
        return EXIT_RUNTIME
    if mode == "nonmonotonic-trace":
        record = {
            "schema_version": 1,
            "record_type": "xvram.pytorch_trace",
            "sequence": 2,
            "monotonic_ns": time.monotonic_ns(),
            "kind": "lease",
            "region_id": 1,
            "allocation_id": 1,
            "operation": "retired",
            "bytes": 0,
            "reason": "test",
        }
        write_frame(
            sys.stdout.buffer,
            MessageType.TRACE,
            {"sequence": 2, "records": [record, dict(record)]},
        )
        return EXIT_RUNTIME
    if mode == "no-trace":
        report["execution"]["trace_records"] = 0
        write_frame(sys.stdout.buffer, MessageType.FINAL, {"sequence": 2, "report": report})
        return 0
    if mode == "progress-mismatch":
        report["graph"]["region_count"] = 2
        report["execution"]["regions_completed"] = 2
        report["execution"]["leases_acquired"] = 2
        report["execution"]["leases_sealed"] = 2
        report["execution"]["leases_retired"] = 2
        report["execution"]["events_recorded"] = 2
        report["execution"]["events_retired"] = 2
        report["execution"]["region_timings_ms"] = [0.1, 0.1]
    record = {
        "schema_version": 1,
        "record_type": "xvram.pytorch_trace",
        "sequence": 1,
        "monotonic_ns": time.monotonic_ns(),
        "kind": "lease",
        "region_id": 1,
        "allocation_id": None,
        "operation": "retired",
        "bytes": 128,
        "reason": "test",
    }
    write_frame(
        sys.stdout.buffer,
        MessageType.TRACE,
        {"sequence": 2, "records": [record]},
    )
    write_frame(sys.stdout.buffer, MessageType.FINAL, {"sequence": 3, "report": report})
    if mode == "final-slow-exit":
        time.sleep(1.0)
    if mode == "final-hang":
        time.sleep(30)
    if mode == "exit-mismatch":
        return EXIT_RUNTIME
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
