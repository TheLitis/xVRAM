from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

from xvram.torch_report import (
    EXIT_COMPLETED,
    EXIT_CORRUPTION,
    EXIT_PREREQUISITE,
    EXIT_PRESSURE,
    EXIT_RUNTIME,
    EXIT_TIMEOUT,
    empty_report,
    finalize_proof,
    success_semantics,
)


def configuration() -> dict[str, object]:
    return {
        "device": 0,
        "model": "llama2-like",
        "model_ratio": 1.522,
        "layers": 31,
        "batch": 1,
        "sequence": 32,
        "hidden": 4096,
        "intermediate": 11008,
        "heads": 32,
        "dtype": "float16",
        "policy": "clock",
        "cache_target_bytes": 7516192768,
        "chunk_size_bytes": 67108864,
        "device_headroom_bytes": 536870912,
        "scratch_cap_bytes": 536870912,
        "prefetch_distance": 2,
        "sdpa_backend": "math",
        "seed": "0x585652414d503034",
    }


def completed() -> dict[str, object]:
    report = empty_report(configuration(), exit_code=EXIT_COMPLETED, status="completed", message="proof completed")
    report["device"]["total_vram_bytes"] = 8589410304
    report["model"].update(
        {
            "logical_bytes": 13072076800,
            "parameter_bytes": 13072064512,
            "activation_bytes": 4096,
        }
    )
    report["graph"].update({"hash": "a" * 64, "backend_hash": "a" * 64, "node_count": 42, "region_count": 31})
    report["execution"].update(
        {
            "leases_acquired": 31,
            "leases_sealed": 31,
            "leases_retired": 31,
            "events_recorded": 31,
            "events_retired": 31,
            "live_views_peak": 8,
            "regions_completed": 31,
            "region_timings_ms": [1.0] * 31,
            "trace_complete": True,
        }
    )
    report["cache"].update(
        {
            "maps": 210,
            "set_access": 210,
            "unmaps": 210,
            "misses": 210,
            "h2d_bytes": 13072064512,
            "evictions": 98,
            "frame_reuses": 98,
        }
    )
    report["verification"].update({"reference_digest": "b" * 64, "output_digest": "b" * 64})
    report["proof"]["stable_addresses"] = True
    for key in report["cleanup"]:
        report["cleanup"][key] = True
    return finalize_proof(report)


def contains_forbidden_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = key.lower()
            if "virtual_address" in lower or "stream_handle" in lower or lower in {"cuda_va", "raw_va"}:
                return True
            if contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_key(child) for child in value)
    return False


def main() -> int:
    report_schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    trace_schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    validate = jsonschema.Draft202012Validator(report_schema).validate
    success = completed()
    validate(success)
    ok, errors = success_semantics(success)
    if not ok:
        raise AssertionError(errors)
    if contains_forbidden_key(success):
        raise AssertionError("report exposes raw CUDA address or stream handle")

    for code, status in (
        (EXIT_PREREQUISITE, "skipped"),
        (EXIT_CORRUPTION, "corruption"),
        (EXIT_PRESSURE, "oom"),
        (EXIT_TIMEOUT, "timeout"),
        (EXIT_RUNTIME, "failed"),
    ):
        validate(empty_report(configuration(), exit_code=code, status=status, message="fixture"))

    trace = {
        "schema_version": 1,
        "record_type": "xvram.pytorch_trace",
        "sequence": 1,
        "monotonic_ns": 123,
        "kind": "lease",
        "region_id": 1,
        "allocation_id": 7,
        "operation": "acquire",
        "bytes": 67108864,
        "reason": "demand",
    }
    jsonschema.Draft202012Validator(trace_schema).validate(trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
