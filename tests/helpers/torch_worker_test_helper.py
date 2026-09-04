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
    if int(report["schema_version"]) == 2:
        report["execution"].update(events_recorded=2, events_retired=2)
        report["compression"].update(
            logical_bytes=1152,
            stored_bytes=1024,
            stored_peak_bytes=1024,
            raw_bytes=512,
            compressed_bytes=512,
            host_budget_bytes=2048,
            host_budget_peak_bytes=4096,
            conversion_scratch_peak_bytes=512,
            logical_h2d_bytes=128,
            pcie_h2d_bytes=96,
            pcie_h2d_payload_bytes=88,
            pcie_h2d_metadata_bytes=8,
            raw_path_decisions=1,
            cpu_lz4_gpu_decode_decisions=1,
            compression_attempts=1,
            compression_commits=1,
            decompression_attempts=1,
            decompression_commits=1,
            generations_created=2,
            generations_committed=2,
            codec_events_recorded=1,
            codec_events_retired=1,
        )
    for key in report["cleanup"]:
        if key not in {"worker_reaped", "complete"}:
            report["cleanup"][key] = True
    return report


def main() -> int:
    mode = sys.argv[1]
    plan_frame = read_frame(sys.stdin.buffer)
    plan = plan_frame.payload
    protocol_version = plan_frame.protocol_version

    def emit(message_type: MessageType, payload: dict[str, object]) -> None:
        write_frame(
            sys.stdout.buffer,
            message_type,
            payload,
            protocol_version=protocol_version,
        )
    if mode == "crash":
        return 19
    if mode == "hang":
        time.sleep(30)
        return 0
    if mode == "heartbeat-hang":
        sequence = 0
        while True:
            sequence += 1
            emit(
                MessageType.HEARTBEAT,
                {"sequence": sequence, "monotonic_ns": time.monotonic_ns()},
            )
            time.sleep(0.05)
    if mode == "truncated":
        magic = b"XVT1\x01" if protocol_version == 1 else b"XVT2\x02"
        sys.stdout.buffer.write(magic + b"\x05\x00\x00\x00\x10{}")
        sys.stdout.buffer.flush()
        return 0
    if mode == "oversized":
        magic = b"XVT1\x01" if protocol_version == 1 else b"XVT2\x02"
        sys.stdout.buffer.write(magic + b"\x05\x00\x10\x00\x01")
        sys.stdout.buffer.flush()
        return 0
    report = completed_report(plan)
    emit(
        MessageType.PROGRESS,
        {"sequence": 1, "operations_retired": 1},
    )
    if mode == "malformed-final":
        report["execution"]["leases_acquired"] = "not-an-integer"
    if mode == "nonmonotonic-protocol":
        emit(
            MessageType.HEARTBEAT,
            {"sequence": 1, "monotonic_ns": time.monotonic_ns()},
        )
        return EXIT_RUNTIME
    if mode == "nonmonotonic-trace":
        record = {
            "schema_version": protocol_version,
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
        if protocol_version == 2:
            record.update(
                chunk_index=0,
                source_generation=1,
                target_generation=2,
                slot_generation=3,
                representation="lz4_blocks",
                codec_path="cpu_lz4_gpu_decode",
            )
        emit(
            MessageType.TRACE,
            {"sequence": 2, "records": [record, dict(record)]},
        )
        return EXIT_RUNTIME
    if mode == "no-trace":
        report["execution"]["trace_records"] = 0
        emit(MessageType.FINAL, {"sequence": 2, "report": report})
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
        "schema_version": protocol_version,
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
    if protocol_version == 2:
        record.update(
            chunk_index=0,
            source_generation=1,
            target_generation=2,
            slot_generation=3,
            representation="lz4_blocks",
            codec_path="cpu_lz4_gpu_decode",
        )
    emit(
        MessageType.TRACE,
        {"sequence": 2, "records": [record]},
    )
    emit(MessageType.FINAL, {"sequence": 3, "report": report})
    if mode == "final-slow-exit":
        time.sleep(1.0)
    if mode == "final-hang":
        time.sleep(30)
    if mode == "exit-mismatch":
        return EXIT_RUNTIME
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
