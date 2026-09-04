#!/usr/bin/env python3
"""Negative and positive causal fixtures for the Phase 5 compression trace."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import jsonschema


def transfer_record(
    event: str,
    sequence: int,
    *,
    allocation_id: int = 7,
    chunk_index: int = 11,
    operation_id: int = 19,
    source_generation: int = 3,
    target_generation: int | None = None,
    slot_generation: int = 5,
    path: str = "raw",
    logical_bytes: int = 64 * 1024**2,
    physical_bytes: int = 64 * 1024**2,
    speculative: bool = False,
) -> dict[str, Any]:
    if event.startswith("encode") or event.startswith("d2h"):
        target_generation = source_generation + 1
    from_representation = "raw" if event.startswith(("h2d", "d2h")) else None
    to_representation = "raw" if event.startswith("d2h") else None
    if event.startswith(("encode", "decode")):
        path = "nvcomp_gpu_codec"
    if event.startswith("decode"):
        from_representation = "lz4_blocks"
    if event.startswith("encode"):
        to_representation = "lz4_blocks"
    return {
        "schema_version": 1,
        "report_type": "xvram.compression_trace",
        "sequence": sequence,
        "monotonic_time_ns": sequence * 100,
        "event": event,
        "allocation_id": allocation_id,
        "chunk_index": chunk_index,
        "operation_id": operation_id,
        "source_generation": source_generation,
        "target_generation": target_generation,
        "slot_generation": slot_generation,
        "from_representation": from_representation,
        "to_representation": to_representation,
        "path": path,
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "reason": f"fixture_{event}",
        "speculative": speculative,
    }


def renumber(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for sequence, record in enumerate(records, start=1):
        record["sequence"] = sequence
        record["monotonic_time_ns"] = sequence * 100
    return records


def completed_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outcome": {"status": "completed"},
        "telemetry": {
            "trace_records_dropped": 0,
            "trace_complete": True,
            "trace_records_emitted": len(records),
        },
        "codec": {
            "gpu_encode_operations": sum(
                record["event"] == "encode_submit" for record in records
            ),
            "gpu_decode_operations": sum(
                record["event"] == "decode_submit" for record in records
            ),
        },
    }


def expect_failure(
    error_type: type[BaseException],
    callback: Callable[[], object],
    expected_text: str,
) -> None:
    try:
        callback()
    except error_type as error:
        if expected_text.lower() not in str(error).lower():
            raise AssertionError(
                f"expected failure containing {expected_text!r}, got {error!r}"
            ) from error
    else:
        raise AssertionError(f"expected {error_type.__name__}: {expected_text}")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compression_trace_contract.py <trace-schema> <validator-module-dir>")
        return 64

    schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validate_schema = jsonschema.Draft202012Validator(schema).validate

    sys.path.insert(0, str(Path(sys.argv[2]).resolve()))
    from validate_phase5_compression_trace import (  # pylint: disable=import-outside-toplevel
        CompressionTraceValidationError,
        validate_causal_trace,
        validate_trace_accounting,
    )

    raw_h2d = [
        transfer_record("h2d_submit", 1),
        transfer_record("h2d_retire", 2),
    ]
    compressed_h2d = [
        transfer_record(
            "h2d_submit", 3, operation_id=20, slot_generation=6,
            path="nvcomp_gpu_codec", physical_bytes=4 * 1024**2,
        ),
        transfer_record(
            "decode_submit", 4, operation_id=20, slot_generation=6,
            physical_bytes=4 * 1024**2,
        ),
        transfer_record(
            "h2d_retire", 5, operation_id=20, slot_generation=6,
            path="nvcomp_gpu_codec", physical_bytes=4 * 1024**2,
        ),
        transfer_record(
            "decode_retire", 6, operation_id=20, slot_generation=6,
            physical_bytes=4 * 1024**2,
        ),
    ]
    gpu_writeback = [
        transfer_record("encode_submit", 7, operation_id=21, slot_generation=7),
        transfer_record("encode_retire", 8, operation_id=21, slot_generation=7),
        transfer_record(
            "d2h_submit", 9, operation_id=21, slot_generation=7,
            path="nvcomp_gpu_codec",
        ),
        transfer_record(
            "d2h_retire", 10, operation_id=21, slot_generation=7,
            path="nvcomp_gpu_codec",
        ),
    ]
    raw_writeback = [
        transfer_record("d2h_submit", 11, operation_id=22, slot_generation=8),
        transfer_record("d2h_retire", 12, operation_id=22, slot_generation=8),
    ]
    valid = renumber(raw_h2d + compressed_h2d + gpu_writeback + raw_writeback)
    for record in valid:
        validate_schema(record)
    report = completed_report(valid)
    summary = validate_trace_accounting(report, valid)
    if summary.gpu_encode_submissions != 1 or summary.gpu_decode_submissions != 1:
        raise AssertionError("positive trace did not reconcile codec submissions")

    decode_before_h2d = [transfer_record("decode_submit", 1)]
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_causal_trace(decode_before_h2d, require_complete=True),
        "not preceded by H2D",
    )

    retire_without_submit = [transfer_record("h2d_retire", 1)]
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_causal_trace(retire_without_submit, require_complete=True),
        "retired without its submission",
    )

    duplicate_in_flight = renumber(
        [
            transfer_record("h2d_submit", 1),
            transfer_record("h2d_submit", 2),
        ]
    )
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_causal_trace(duplicate_in_flight, require_complete=True),
        "duplicate/in-flight",
    )

    reused_after_retirement = renumber(
        [
            transfer_record("h2d_submit", 1),
            transfer_record("h2d_retire", 2),
            transfer_record("h2d_submit", 3),
            transfer_record("h2d_retire", 4),
        ]
    )
    # The trace schema does not expose a runtime/pipeline epoch. Independent
    # suite cases may therefore restart every numeric identifier after the first
    # case has fully retired, which is safe and must remain accepted.
    validate_causal_trace(reused_after_retirement, require_complete=True)

    stale_generation_retirement = renumber(
        [
            transfer_record("h2d_submit", 1, source_generation=3),
            transfer_record("h2d_retire", 2, source_generation=4),
        ]
    )
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_causal_trace(
            stale_generation_retirement, require_complete=True
        ),
        "stale/mismatched generation",
    )

    incomplete_chain = [transfer_record("encode_submit", 1)]
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_trace_accounting(
            completed_report(incomplete_chain), incomplete_chain
        ),
        "unretired transfer generations",
    )

    mismatched_count_report = completed_report(valid)
    mismatched_count_report["telemetry"]["trace_records_emitted"] += 1
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_trace_accounting(mismatched_count_report, valid),
        "record count does not reconcile",
    )

    mismatched_codec_report = copy.deepcopy(report)
    mismatched_codec_report["codec"]["gpu_decode_operations"] += 1
    expect_failure(
        CompressionTraceValidationError,
        lambda: validate_trace_accounting(mismatched_codec_report, valid),
        "GPU decode trace submissions do not reconcile",
    )

    # A failed/timeout worker may end with a valid partial chain; it must still
    # obey causal ordering, but completeness is required only for completed runs.
    partial_report = completed_report(incomplete_chain)
    partial_report["outcome"]["status"] = "timeout"
    validate_trace_accounting(partial_report, incomplete_chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
