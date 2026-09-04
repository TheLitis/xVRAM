#!/usr/bin/env python3
"""Strict report/trace checks for the Phase 5 PyTorch RTX 3070 gate."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path
from typing import Any

import jsonschema


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = {
    "cuda_va",
    "raw_va",
    "virtual_address",
    "virtual_address_base",
    "native_stream_handle",
    "stream_handle",
    "device_pointer",
    "logical_address",
    "logical_base",
    "cuda_address",
}
_FORBIDDEN_TEXT = re.compile(
    r"\b(?:raw_va|cuda_va|virtual_address|native_stream_handle|stream_handle|"
    r"device_pointer|logical_address|logical_base|cuda_address)\b",
    re.IGNORECASE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _decode_json(value: str) -> dict[str, Any]:
    decoded = base64.b64decode(value, validate=True).decode("utf-8")
    result = json.loads(decoded)
    if not isinstance(result, dict):
        raise ValueError("encoded acceptance metadata must be an object")
    return result


def _assert_private(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            _require(
                lowered not in _FORBIDDEN_KEYS and "virtual_address" not in lowered,
                f"{path} exposes forbidden field {key}",
            )
            _assert_private(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_private(child, f"{path}[{index}]")
    elif isinstance(value, str):
        _require(
            _FORBIDDEN_TEXT.search(value) is None,
            f"{path} exposes a raw address or stream handle",
        )


def _load_trace(path: Path, schema: dict[str, Any]) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    _require(raw.endswith("\n"), "trace JSONL is not newline terminated")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    _require(bool(records), "trace contains no records")
    validator = jsonschema.Draft202012Validator(schema)
    for record in records:
        validator.validate(record)
    sequences = [record["sequence"] for record in records]
    _require(sequences[0] == 1, "trace sequence does not start at one")
    _require(
        all(left < right for left, right in zip(sequences, sequences[1:])),
        "trace sequence is not strictly increasing",
    )
    timestamps = [record["monotonic_ns"] for record in records]
    _require(
        all(left <= right for left, right in zip(timestamps, timestamps[1:])),
        "trace monotonic timestamps moved backwards",
    )
    terminal = records[-1]
    _require(
        terminal["kind"] == "verification"
        and terminal["operation"] == "trace_complete",
        "trace has no terminal completion marker",
    )
    _require(any(record["kind"] == "lease" for record in records), "trace has no lease")
    _require(
        any(
            record["kind"] in {"compression", "codec", "generation"}
            for record in records
        ),
        "v2 trace has no compression/generation observation",
    )
    return records


def _validate_device_budget(report: dict[str, Any]) -> None:
    configuration = report["configuration"]
    cache = report["cache"]
    compression = report["compression"]
    # The frozen native cache_target_bytes counter is the aggregate managed
    # device budget after headroom, not merely the physical-frame allowance.
    # Scratch and codec resources are carved out of it by Runtime::setup.
    target = cache["target_bytes"]
    physical_ceiling = (
        report["device"]["total_vram_bytes"]
        - configuration["device_headroom_bytes"]
    )
    _require(
        target <= physical_ceiling,
        "aggregate cache/compute/codec target plus headroom exceeds VRAM",
    )
    # target is the final live target, while resident_peak is historical. A
    # valid shrink must not be compared against a peak from an earlier budget
    # epoch. The report has no synchronized per-epoch residency/reserve sample;
    # use the global physical ceiling without claiming a live-budget proof.
    managed_charge = (
        cache["resident_peak_bytes"]
        + configuration["scratch_cap_bytes"]
        + compression["workspace_peak_bytes"]
        + compression["slot_peak_bytes"]
    )
    _require(
        managed_charge <= physical_ceiling,
        "resident frame peak plus compute/codec reserves exceeds the physical ceiling",
    )


def validate(
    report_path: Path,
    trace_path: Path,
    report_schema_path: Path,
    trace_schema_path: Path,
    expected: dict[str, Any],
    hardware: dict[str, Any],
) -> None:
    report_schema = json.loads(report_schema_path.read_text(encoding="utf-8-sig"))
    trace_schema = json.loads(trace_schema_path.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.Draft202012Validator.check_schema(trace_schema)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator(report_schema).validate(report)
    records = _load_trace(trace_path, trace_schema)
    _assert_private(report)
    _assert_private(records)

    _require(report["schema_version"] == 2, "report did not use schema v2")
    _require(
        report["report_type"] == "xvram.pytorch_inference",
        "wrong report type",
    )
    _require(
        report["outcome"]["status"] == "completed"
        and report["outcome"]["exit_code"] == 0,
        "run did not complete",
    )
    _require(not report["diagnostics"], "completed report contains diagnostics")
    _require(report["system"]["os"] == "Windows", "gate did not run on Windows")
    _require(
        report["build"]["torch_version"] == hardware["torch_version"],
        "report PyTorch version differs from preflight",
    )
    _require(
        report["build"]["cuda_version"] == hardware["cuda_version"],
        "report CUDA version differs from preflight",
    )
    _require(report["build"]["bridge_api_version"] == 2, "bridge did not use ABI v2")

    device = report["device"]
    _require(device["ordinal"] == expected["device"], "wrong CUDA device ordinal")
    expected_name = (
        hardware["device_name"] if expected["include_identifiers"] else "redacted"
    )
    _require(device["name"] == expected_name, "device identifier redaction differs")
    _require(
        device["total_vram_bytes"] == hardware["total_vram_bytes"],
        "reported VRAM differs from preflight",
    )
    _require(
        device["compute_capability"] == hardware["compute_capability"],
        "reported compute capability differs from preflight",
    )
    _require(
        device["identifiers_included"] == expected["include_identifiers"],
        "device identifier privacy flag differs from the CLI request",
    )

    configuration = report["configuration"]
    fixed_configuration = {
        "device": expected["device"],
        "model": expected["model"],
        "layers": expected["layers"],
        "batch": 1,
        "sequence": expected["sequence"],
        "hidden": 4096,
        "intermediate": 11008,
        "heads": 32,
        "dtype": "float16",
        "policy": expected["policy"],
        "cache_target_bytes": 0,
        "chunk_size_bytes": 64 * 1024**2,
        "device_headroom_bytes": 512 * 1024**2,
        "scratch_cap_bytes": 512 * 1024**2,
        "prefetch_distance": expected["prefetch_distance"],
        "sdpa_backend": "math",
        "seed": "0x585652414d503035",
        "compression": expected["compression"],
        "compression_codec": "lz4",
        "state_pattern": expected["state_pattern"],
        "host_store_cap_bytes": 0,
        "host_headroom_bytes": 0,
        "compression_scratch_cap_bytes": 256 * 1024**2,
        "codec_slots": 2,
        "codec_workers": 2,
        "include_identifiers": expected["include_identifiers"],
    }
    for key, value in fixed_configuration.items():
        _require(configuration.get(key) == value, f"configuration.{key} differs")
    _require(
        math.isclose(
            float(configuration["model_ratio"]),
            float(expected["model_ratio"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "configuration.model_ratio differs",
    )

    model = report["model"]
    for name in ("layers", "batch", "sequence", "hidden", "intermediate", "heads"):
        _require(model[name] == fixed_configuration[name], f"model.{name} differs")
    _require(model["kind"] == expected["model"], "model kind differs")
    for name in ("parameter_bytes", "activation_bytes", "logical_bytes"):
        _require(model[name] == expected[name], f"model.{name} differs")
    if expected["model"] == "llama2-like":
        observed_ratio = model["parameter_bytes"] / hardware["total_vram_bytes"]
        _require(
            math.isclose(
                observed_ratio,
                float(expected["model_ratio"]),
                rel_tol=0.0,
                abs_tol=0.006,
            ),
            f"observed model ratio {observed_ratio:.9f} differs from the gate",
        )
    oversubscribed = model["logical_bytes"] > device["total_vram_bytes"]
    _require(oversubscribed == expected["oversubscribed"], "oversubscription differs")

    graph = report["graph"]
    _require(_SHA256.fullmatch(graph["hash"]) is not None, "invalid graph hash")
    _require(graph["backend_hash"] == graph["hash"], "backend graph hash differs")
    _require(
        _SHA256.fullmatch(graph["allowlist_hash"]) is not None,
        "invalid allowlist hash",
    )
    _require(graph["node_count"] > 0 and graph["region_count"] > 0, "empty graph")
    _require(not graph["unsupported_nodes"], "graph contains unsupported nodes")

    execution = report["execution"]
    leases = (
        execution["leases_acquired"],
        execution["leases_sealed"],
        execution["leases_retired"],
    )
    _require(leases[0] == leases[1] == leases[2], "lease lifecycle does not reconcile")
    _require(
        execution["events_recorded"] > 0
        and execution["events_recorded"] == execution["events_retired"],
        "event lifecycle does not reconcile",
    )
    _require(
        execution["regions_completed"] == graph["region_count"],
        "graph regions did not all retire",
    )
    _require(
        len(execution["region_timings_ms"]) == execution["regions_completed"]
        and all(
            math.isfinite(value) and value >= 0
            for value in execution["region_timings_ms"]
        ),
        "region timings are incomplete",
    )
    _require(execution["live_views_final"] == 0, "managed views leaked")
    _require(
        execution["scratch_peak_bytes"] <= execution["scratch_cap_bytes"],
        "compute scratch exceeded its cap",
    )
    _require(
        execution["prefetch_distance"] == expected["prefetch_distance"],
        "execution prefetch distance differs",
    )
    _require(
        execution["trace_complete"]
        and execution["trace_records"] == len(records),
        "trace accounting does not reconcile",
    )

    cache = report["cache"]
    _require(cache["chunk_bytes"] == 64 * 1024**2, "cache chunk size differs")
    _require(
        cache["target_bytes"] > 0,
        "cache target was not observed",
    )
    if oversubscribed:
        _require(
            cache["target_bytes"] < model["logical_bytes"],
            "cache target does not prove bounded oversubscription",
        )
    _require(
        cache["resident_bytes"] == 0 and cache["resident_peak_bytes"] > 0,
        "cache residency was not observed and drained",
    )
    _require(
        cache["maps"] > 0
        and cache["maps"] == cache["set_access"] == cache["unmaps"],
        "map/SetAccess/unmap lifecycle does not reconcile",
    )
    _require(
        cache["unsafe_remaps"] == 0 and cache["unsafe_transitions"] == 0,
        "unsafe cache activity was observed",
    )
    _require(cache["weight_d2h_bytes"] == 0, "read-only weights were written back")
    _require(
        cache["prefetch_submitted"] == cache["prefetch_retired"],
        "prefetch lifecycle does not reconcile",
    )
    if expected["prefetch_distance"] == 0:
        _require(cache["prefetch_submitted"] == 0, "distance-zero run prefetched")
    else:
        _require(cache["prefetch_submitted"] > 0, "prefetch-enabled run did not prefetch")
    if oversubscribed:
        _require(
            cache["evictions"] > 0 and cache["frame_reuses"] > 0,
            "oversubscribed run observed no eviction/frame reuse",
        )

    compression = report["compression"]
    _require(compression["policy"] == expected["compression"], "compression policy differs")
    _require(compression["codec"] == "lz4", "compression codec differs")
    _require(
        compression["logical_bytes"] == model["logical_bytes"],
        "compression logical bytes do not reconcile",
    )
    _require(
        compression["stored_bytes"]
        == compression["raw_bytes"] + compression["compressed_bytes"],
        "host representations do not reconcile",
    )
    _require(
        0 < compression["host_store_cap_bytes"]
        and 0 < compression["host_headroom_bytes"],
        "effective host budget was not observed",
    )
    _require(
        compression["host_budget_peak_bytes"] <= compression["host_store_cap_bytes"]
        and compression["stored_peak_bytes"] <= compression["host_budget_peak_bytes"],
        "host storage exceeded its effective cap",
    )
    _require(
        compression["host_budget_bytes"] <= compression["host_budget_peak_bytes"],
        "current host budget exceeds its peak",
    )
    _require(
        compression["workspace_peak_bytes"]
        <= configuration["compression_scratch_cap_bytes"],
        "codec workspace exceeded its cap",
    )
    _require(
        compression["codec_events_recorded"]
        == compression["codec_events_retired"],
        "codec event lifecycle does not reconcile",
    )
    _require(
        compression["generations_created"]
        == compression["generations_committed"]
        + compression["generations_discarded"],
        "backing generations do not reconcile",
    )
    _require(
        compression["logical_h2d_bytes"] == cache["h2d_bytes"]
        and compression["logical_d2h_bytes"]
        == cache["d2h_bytes"]
        + compression["rejected_candidate_logical_d2h_bytes"],
        "cache and compression transfer bytes do not reconcile",
    )
    _require(
        compression["pcie_h2d_bytes"]
        == compression["pcie_h2d_payload_bytes"]
        + compression["pcie_h2d_metadata_bytes"]
        and compression["pcie_d2h_bytes"]
        == compression["pcie_d2h_payload_bytes"]
        + compression["pcie_d2h_metadata_bytes"],
        "PCIe payload and metadata bytes do not reconcile with transfer totals",
    )
    _require(
        compression["pcie_h2d_payload_bytes"] <= compression["logical_h2d_bytes"]
        and compression["pcie_d2h_payload_bytes"] <= compression["logical_d2h_bytes"]
        and compression["rejected_candidate_logical_d2h_bytes"]
        <= compression["logical_d2h_bytes"],
        "PCIe payload or rejected-candidate bytes exceed logical transfers",
    )
    _require(
        compression["hot_allocation_d2h_bytes"] == 0
        and compression["non_hot_allocation_d2h_bytes"]
        == cache["d2h_bytes"],
        "read-only weight D2H accounting is invalid",
    )
    _validate_device_budget(report)

    compressed_decisions = (
        compression["cpu_lz4_gpu_decode_decisions"]
        + compression["gpu_lz4_decisions"]
    )
    codec_records = [record for record in records if record["codec_path"] is not None]
    _require(bool(codec_records), "trace contains no codec-path observation")
    if expected["state_pattern"] == "incompressible":
        _require(
            compression["raw_path_decisions"] > compressed_decisions,
            "adaptive incompressible state did not predominantly select raw",
        )
        _require(
            any(record["codec_path"] == "raw" for record in codec_records),
            "raw path was not observed in the trace",
        )
    else:
        _require(compressed_decisions > 0, "structured state made no compressed decision")
        _require(compression["compression_commits"] > 0, "no compressed generation committed")
        _require(compression["decompression_commits"] > 0, "no compressed generation decoded")
        _require(
            compression["pcie_h2d_bytes"] < compression["logical_h2d_bytes"],
            "structured state did not reduce H2D traffic",
        )
        _require(
            compression["stored_peak_bytes"] < compression["logical_bytes"],
            "structured state did not reduce peak host storage",
        )
        _require(
            any(
                record["representation"] == "lz4_blocks"
                or record["codec_path"]
                in {"cpu_lz4_gpu_decode", "nvcomp_gpu_codec"}
                for record in records
            ),
            "trace contains no compressed representation/path",
        )

    verification = report["verification"]
    _require(
        _SHA256.fullmatch(verification["reference_digest"]) is not None
        and verification["reference_digest"] == verification["output_digest"],
        "managed/reference digests differ",
    )
    _require(
        verification["mismatch_count"] == 0
        and verification["max_abs_error"] == 0
        and verification["max_rel_error"] == 0,
        "managed output is not exactly equal to the streamed reference",
    )
    proof = report["proof"]
    for name in (
        "stable_addresses",
        "maps_match_set_access",
        "event_safe",
        "zero_weight_writeback",
        "bounded_scratch",
        "reference_matches",
        "zero_live_views",
        "full_cleanup",
    ):
        _require(proof[name] is True, f"proof.{name} is false")
    _require(
        proof["logical_exceeds_vram"] == expected["oversubscribed"],
        "proof.logical_exceeds_vram differs from observed bytes",
    )
    _require(
        proof["real_oversubscription"] == expected["oversubscribed"],
        "proof.real_oversubscription differs from the gate",
    )
    for name, value in report["cleanup"].items():
        _require(value is True, f"cleanup.{name} is false")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report-schema", type=Path, required=True)
    parser.add_argument("--trace-schema", type=Path, required=True)
    parser.add_argument("--expected-base64", required=True)
    parser.add_argument("--hardware-base64", required=True)
    args = parser.parse_args()
    validate(
        args.report,
        args.trace,
        args.report_schema,
        args.trace_schema,
        _decode_json(args.expected_base64),
        _decode_json(args.hardware_base64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
