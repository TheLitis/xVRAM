#!/usr/bin/env python3
"""Validate the Phase 5 report/JSONL trace contract and causal transfer chains."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jsonschema


TRANSFER_EVENTS = frozenset(
    {
        "h2d_submit",
        "h2d_retire",
        "decode_submit",
        "decode_retire",
        "encode_submit",
        "encode_retire",
        "d2h_submit",
        "d2h_retire",
    }
)
FORBIDDEN_KEY_PARTS = (
    "virtual_address",
    "stream_handle",
    "device_pointer",
    "host_pointer",
    "cuda_va",
    "raw_va",
    "cuda_address",
    "logical_base",
)
SAFE_PROOF_KEYS = {
    "stable_virtual_addresses_verified",
    "raw_virtual_addresses_omitted",
}
POINTER_TEXT = re.compile(r"^0x[0-9a-fA-F]{8,}$")


class CompressionTraceValidationError(AssertionError):
    """A schema-valid trace violates ordering, accounting, or privacy."""


@dataclass(frozen=True)
class CausalTraceSummary:
    record_count: int
    gpu_encode_submissions: int
    gpu_decode_submissions: int


@dataclass(frozen=True)
class _Chain:
    state: str
    logical_bytes: int
    target_generation: int | None
    speculative: bool | None


TraceIdentity = tuple[int, int, int, int, int, str]


def check_private_runtime_values(value: object, path: str = "root") -> None:
    """Reject CUDA addresses, native pointers, and stream handles from public artifacts."""

    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered not in SAFE_PROOF_KEYS and any(
                part in lowered for part in FORBIDDEN_KEY_PARTS
            ):
                raise CompressionTraceValidationError(
                    f"{path} exposes forbidden key {key}"
                )
            check_private_runtime_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_private_runtime_values(child, f"{path}[{index}]")
    elif isinstance(value, str) and POINTER_TEXT.fullmatch(value):
        raise CompressionTraceValidationError(f"{path} exposes pointer-like text")


def _required_identity(record: Mapping[str, Any], index: int) -> TraceIdentity:
    fields = (
        "allocation_id",
        "chunk_index",
        "operation_id",
        "source_generation",
        "slot_generation",
        "path",
    )
    values = tuple(record.get(field) for field in fields)
    if (
        any(value is None for value in values)
        or any(not isinstance(value, int) for value in values[:5])
        or not isinstance(values[5], str)
    ):
        raise CompressionTraceValidationError(
            f"transfer identity is incomplete at trace record {index}"
        )
    return values  # type: ignore[return-value]


def _chain_from(record: Mapping[str, Any], state: str) -> _Chain:
    return _Chain(
        state=state,
        logical_bytes=record["logical_bytes"],
        target_generation=record.get("target_generation"),
        speculative=record.get("speculative"),
    )


def _continue_chain(
    chain: _Chain,
    record: Mapping[str, Any],
    state: str,
    index: int,
) -> _Chain:
    if record["logical_bytes"] != chain.logical_bytes:
        raise CompressionTraceValidationError(
            f"logical byte count changed within a transfer chain at record {index}"
        )
    if record.get("speculative") != chain.speculative:
        raise CompressionTraceValidationError(
            f"speculative flag changed within a transfer chain at record {index}"
        )
    target_generation = record.get("target_generation")
    if chain.target_generation is not None and target_generation != chain.target_generation:
        raise CompressionTraceValidationError(
            f"target generation changed within a transfer chain at record {index}"
        )
    return _Chain(
        state=state,
        logical_bytes=chain.logical_bytes,
        target_generation=(
            chain.target_generation
            if chain.target_generation is not None
            else target_generation
        ),
        speculative=chain.speculative,
    )


def validate_causal_trace(
    records: Sequence[Mapping[str, Any]], *, require_complete: bool
) -> CausalTraceSummary:
    """Validate event-safe H2D/decode and encode/D2H generations.

    A chain identity contains both the operation and slot generation. The public
    trace has no runtime/pipeline epoch, so an identical tuple may legitimately
    reappear after a chain fully retires in a later independent runtime. Concurrent
    reuse and retirements with observably mismatched generations remain invalid.
    """

    active: dict[TraceIdentity, _Chain] = {}
    gpu_encode_submissions = 0
    gpu_decode_submissions = 0

    def begin(
        family: str,
        identity: TraceIdentity,
        record: Mapping[str, Any],
        state: str,
        index: int,
    ) -> None:
        if identity in active:
            raise CompressionTraceValidationError(
                f"duplicate/in-flight {family} submission at trace record {index}"
            )
        active[identity] = _chain_from(record, state)

    def has_mismatched_active_generation(identity: TraceIdentity) -> bool:
        return any(
            candidate[:3] == identity[:3] and candidate[5] == identity[5]
            for candidate in active
        )

    for index, record in enumerate(records):
        event = record.get("event")
        if event not in TRANSFER_EVENTS:
            continue
        identity = _required_identity(record, index)
        chain = active.get(identity)

        if event == "h2d_submit":
            begin("H2D", identity, record, "h2d_submitted", index)
        elif event == "decode_submit":
            if chain is None or chain.state != "h2d_submitted":
                raise CompressionTraceValidationError(
                    f"decode was not preceded by H2D submission at record {index}"
                )
            active[identity] = _continue_chain(
                chain, record, "decode_submitted", index
            )
            gpu_decode_submissions += 1
        elif event == "h2d_retire":
            if chain is None:
                if has_mismatched_active_generation(identity):
                    raise CompressionTraceValidationError(
                        f"H2D retirement has a stale/mismatched generation at record {index}"
                    )
                raise CompressionTraceValidationError(
                    f"H2D retired without its submission at record {index}"
                )
            if chain.state == "h2d_submitted":
                _continue_chain(chain, record, "h2d_retired", index)
                del active[identity]
            elif chain.state == "decode_submitted":
                active[identity] = _continue_chain(
                    chain, record, "decode_h2d_retired", index
                )
            else:
                raise CompressionTraceValidationError(
                    f"H2D retired without its submission at record {index}"
                )
        elif event == "decode_retire":
            if chain is None or chain.state != "decode_h2d_retired":
                if chain is None and has_mismatched_active_generation(identity):
                    raise CompressionTraceValidationError(
                        f"decode retirement has a stale/mismatched generation at record {index}"
                    )
                raise CompressionTraceValidationError(
                    f"decode retired before H2D/event proof at record {index}"
                )
            _continue_chain(chain, record, "decode_retired", index)
            del active[identity]
        elif event == "encode_submit":
            begin("encode", identity, record, "encode_submitted", index)
            gpu_encode_submissions += 1
        elif event == "encode_retire":
            if chain is None or chain.state != "encode_submitted":
                if chain is None and has_mismatched_active_generation(identity):
                    raise CompressionTraceValidationError(
                        f"encode retirement has a stale/mismatched generation at record {index}"
                    )
                raise CompressionTraceValidationError(
                    f"encode retired without submission at record {index}"
                )
            active[identity] = _continue_chain(
                chain, record, "encode_retired", index
            )
        elif event == "d2h_submit":
            if chain is None:
                begin("D2H", identity, record, "d2h_submitted", index)
            elif chain.state == "encode_retired":
                active[identity] = _continue_chain(
                    chain, record, "encode_d2h_submitted", index
                )
            else:
                raise CompressionTraceValidationError(
                    f"D2H submitted before safe encode state at record {index}"
                )
        elif event == "d2h_retire":
            if chain is None or chain.state not in {
                "d2h_submitted",
                "encode_d2h_submitted",
            }:
                if chain is None and has_mismatched_active_generation(identity):
                    raise CompressionTraceValidationError(
                        f"D2H retirement has a stale/mismatched generation at record {index}"
                    )
                raise CompressionTraceValidationError(
                    f"D2H retired without submission/host commit at record {index}"
                )
            _continue_chain(chain, record, "d2h_retired", index)
            del active[identity]

    if require_complete and active:
        descriptions = ", ".join(
            f"{identity}:{chain.state}"
            for identity, chain in sorted(active.items())[:3]
        )
        raise CompressionTraceValidationError(
            f"completed trace has {len(active)} unretired transfer generations: "
            f"{descriptions}"
        )

    return CausalTraceSummary(
        record_count=len(records),
        gpu_encode_submissions=gpu_encode_submissions,
        gpu_decode_submissions=gpu_decode_submissions,
    )


def validate_trace_accounting(
    report: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> CausalTraceSummary:
    """Validate causal chains and completed-report trace reconciliation."""

    completed = report["outcome"]["status"] == "completed"
    summary = validate_causal_trace(records, require_complete=completed)
    if not completed:
        return summary

    telemetry = report["telemetry"]
    if telemetry["trace_records_dropped"] != 0 or not telemetry["trace_complete"]:
        raise CompressionTraceValidationError(
            "completed trace accounting is incomplete"
        )
    if telemetry["trace_records_emitted"] != len(records):
        raise CompressionTraceValidationError("trace record count does not reconcile")
    codec = report["codec"]
    if summary.gpu_encode_submissions != codec["gpu_encode_operations"]:
        raise CompressionTraceValidationError(
            "GPU encode trace submissions do not reconcile"
        )
    if summary.gpu_decode_submissions != codec["gpu_decode_operations"]:
        raise CompressionTraceValidationError(
            "GPU decode trace submissions do not reconcile"
        )
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw_trace = path.read_text(encoding="utf-8-sig")
    if not raw_trace.endswith("\n"):
        raise CompressionTraceValidationError("trace is not newline terminated")
    records = [json.loads(line) for line in raw_trace.splitlines() if line.strip()]
    if not records:
        raise CompressionTraceValidationError("trace contains no records")
    return records


def validate_artifacts(
    report_schema_path: Path,
    trace_schema_path: Path,
    report_path: Path,
    trace_path: Path | None,
    *,
    allow_missing_trace: bool,
) -> None:
    """Validate the schemas, privacy, JSONL ordering, causality, and accounting."""

    report_schema = json.loads(report_schema_path.read_text(encoding="utf-8-sig"))
    trace_schema = json.loads(trace_schema_path.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.Draft202012Validator.check_schema(trace_schema)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator(report_schema).validate(report)
    check_private_runtime_values(report)

    if trace_path is None or not trace_path.is_file() or trace_path.stat().st_size == 0:
        if allow_missing_trace:
            return
        raise CompressionTraceValidationError("trace was not created or is empty")

    records = read_jsonl(trace_path)
    trace_validator = jsonschema.Draft202012Validator(trace_schema)
    for record in records:
        trace_validator.validate(record)
        check_private_runtime_values(record, "trace")

    sequences = [record["sequence"] for record in records]
    if sequences[0] != 1 or any(
        left >= right for left, right in zip(sequences, sequences[1:])
    ):
        raise CompressionTraceValidationError(
            "trace sequences are not strictly increasing from one"
        )
    timestamps = [record["monotonic_time_ns"] for record in records]
    if any(left > right for left, right in zip(timestamps, timestamps[1:])):
        raise CompressionTraceValidationError(
            "trace monotonic timestamps moved backwards"
        )
    validate_trace_accounting(report, records)


def _parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-schema", required=True, type=Path)
    parser.add_argument("--trace-schema", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--allow-missing-trace", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    options = _parse_args(arguments)
    validate_artifacts(
        options.report_schema,
        options.trace_schema,
        options.report,
        options.trace,
        allow_missing_trace=options.allow_missing_trace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
