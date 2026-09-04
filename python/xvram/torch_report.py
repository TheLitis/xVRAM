"""Strict report construction and semantic checks for Phase 4b."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
from typing import Any, Mapping

from .torch_planner import PLANNER_VERSION, operator_allowlist_hash, supported_operator_targets


REPORT_TYPE = "xvram.pytorch_inference"
SCHEMA_VERSION = 1
SCHEMA_VERSION_COMPRESSION = 2
ALLOWLIST_VERSION = PLANNER_VERSION
ALLOWLIST = supported_operator_targets()
ALLOWLIST_HASH = operator_allowlist_hash()


EXIT_COMPLETED = 0
EXIT_PREREQUISITE = 23
EXIT_CORRUPTION = 24
EXIT_PRESSURE = 25
EXIT_TIMEOUT = 26
EXIT_RUNTIME = 27
EXIT_USAGE = 64
EXIT_INTERNAL = 70
EXIT_OUTPUT = 74

_TOP_LEVEL_KEYS = {
    "schema_version",
    "report_type",
    "build",
    "system",
    "device",
    "configuration",
    "model",
    "graph",
    "execution",
    "cache",
    "verification",
    "proof",
    "outcome",
    "cleanup",
    "diagnostics",
}
_TOP_LEVEL_KEYS_V2 = _TOP_LEVEL_KEYS | {"compression"}
_TRACE_KEYS = {
    "schema_version",
    "record_type",
    "sequence",
    "monotonic_ns",
    "kind",
    "region_id",
    "allocation_id",
    "operation",
    "bytes",
    "reason",
}
_TRACE_KEYS_V2 = _TRACE_KEYS | {
    "chunk_index",
    "source_generation",
    "target_generation",
    "slot_generation",
    "representation",
    "codec_path",
}

_SECTION_KEYS = {
    "build": {
        "version",
        "git_commit",
        "torch_version",
        "cuda_version",
        "torch_target_version",
        "bridge_api_version",
    },
    "system": {"os", "architecture", "python_version"},
    "device": {
        "ordinal",
        "name",
        "total_vram_bytes",
        "compute_capability",
        "identifiers_included",
    },
    "model": {
        "kind",
        "layers",
        "batch",
        "sequence",
        "hidden",
        "intermediate",
        "heads",
        "logical_bytes",
        "parameter_bytes",
        "activation_bytes",
    },
    "graph": {
        "capture",
        "hash",
        "backend_hash",
        "allowlist_version",
        "allowlist_hash",
        "node_count",
        "region_count",
        "unsupported_nodes",
    },
    "execution": {
        "leases_acquired",
        "leases_sealed",
        "leases_retired",
        "tiled_gemm_regions",
        "tiled_gemm_tiles",
        "events_recorded",
        "events_retired",
        "live_views_peak",
        "live_views_final",
        "regions_completed",
        "region_timings_ms",
        "scratch_peak_bytes",
        "scratch_cap_bytes",
        "prefetch_distance",
        "trace_records",
        "trace_complete",
    },
    "cache": {
        "target_bytes",
        "chunk_bytes",
        "resident_bytes",
        "resident_peak_bytes",
        "maps",
        "set_access",
        "unmaps",
        "hits",
        "misses",
        "h2d_bytes",
        "d2h_bytes",
        "weight_d2h_bytes",
        "evictions",
        "frame_reuses",
        "prefetch_submitted",
        "prefetch_retired",
        "unsafe_remaps",
        "unsafe_transitions",
    },
    "verification": {
        "reference_digest",
        "output_digest",
        "max_abs_error",
        "max_rel_error",
        "mismatch_count",
    },
    "proof": {
        "logical_exceeds_vram",
        "real_oversubscription",
        "stable_addresses",
        "maps_match_set_access",
        "event_safe",
        "zero_weight_writeback",
        "bounded_scratch",
        "reference_matches",
        "zero_live_views",
        "full_cleanup",
    },
    "outcome": {"status", "exit_code", "message"},
    "cleanup": {
        "worker_reaped",
        "leases_drained",
        "views_released",
        "mappings_unmapped",
        "handles_released",
        "reservations_released",
        "streams_destroyed",
        "context_released",
        "complete",
    },
}
_COMPRESSION_SECTION_KEYS = {
    "policy",
    "codec",
    "host_store_cap_bytes",
    "host_headroom_bytes",
    "host_budget_bytes",
    "host_budget_peak_bytes",
    "conversion_scratch_peak_bytes",
    "logical_bytes",
    "stored_bytes",
    "stored_peak_bytes",
    "raw_bytes",
    "compressed_bytes",
    "implicit_zero_bytes",
    "invalid_bytes",
    "logical_h2d_bytes",
    "pcie_h2d_bytes",
    "pcie_h2d_payload_bytes",
    "pcie_h2d_metadata_bytes",
    "logical_d2h_bytes",
    "pcie_d2h_bytes",
    "pcie_d2h_payload_bytes",
    "pcie_d2h_metadata_bytes",
    "rejected_candidate_logical_d2h_bytes",
    "hot_allocation_d2h_bytes",
    "non_hot_allocation_d2h_bytes",
    "raw_path_decisions",
    "cpu_lz4_gpu_decode_decisions",
    "gpu_lz4_decisions",
    "never_compress_decisions",
    "raw_fallbacks",
    "cpu_codec_fallbacks",
    "gpu_codec_fallbacks",
    "compression_attempts",
    "compression_commits",
    "decompression_attempts",
    "decompression_commits",
    "generations_created",
    "generations_committed",
    "generations_discarded",
    "workspace_bytes",
    "workspace_peak_bytes",
    "slot_bytes",
    "slot_peak_bytes",
    "spill_reserved_bytes",
    "spill_reserved_peak_bytes",
    "cpu_encode_nanoseconds",
    "cpu_decode_nanoseconds",
    "gpu_encode_nanoseconds",
    "gpu_decode_nanoseconds",
    "verification_nanoseconds",
    "codec_events_recorded",
    "codec_events_retired",
}
_CONFIGURATION_REQUIRED_KEYS = {
    "device",
    "model",
    "model_ratio",
    "layers",
    "batch",
    "sequence",
    "hidden",
    "intermediate",
    "heads",
    "dtype",
    "policy",
    "cache_target_bytes",
    "chunk_size_bytes",
    "device_headroom_bytes",
    "scratch_cap_bytes",
    "prefetch_distance",
    "sdpa_backend",
    "seed",
}
_CONFIGURATION_OPTIONAL_KEYS = {"include_identifiers"}
_CONFIGURATION_V2_KEYS = {
    "compression",
    "compression_codec",
    "state_pattern",
    "host_store_cap_bytes",
    "host_headroom_bytes",
    "compression_scratch_cap_bytes",
    "codec_slots",
    "codec_workers",
}
_DIAGNOSTIC_KEYS = {"stage", "code", "message"}
_ALLOWED_EXIT_CODES = {
    EXIT_COMPLETED,
    EXIT_PREREQUISITE,
    EXIT_CORRUPTION,
    EXIT_PRESSURE,
    EXIT_TIMEOUT,
    EXIT_RUNTIME,
    EXIT_USAGE,
    EXIT_INTERNAL,
    EXIT_OUTPUT,
}
_TRACE_KINDS = {"lease", "transition", "prefetch", "discard", "scratch", "verification"}
_TRACE_KINDS_V2 = _TRACE_KINDS | {"compression", "codec", "generation"}
_FORBIDDEN_IDENTITY_TEXT = re.compile(
    r"(?:"
    r"(?:cuda[ _-]?va|raw[ _-]?va|virtual[ _-]?address|(?:device[ _-]?|native[ _-]?|raw[ _-]?)?pointer|(?:native[ _-]?)?stream[ _-]?handle)"
    r"\s*(?:=|:)?\s*(?:0x[0-9a-f]+|[0-9]{4,})"
    r")",
    re.IGNORECASE,
)
_BARE_ADDRESS_TEXT = re.compile(r"(?<![0-9a-f])0x[0-9a-f]{8,}(?![0-9a-f])", re.IGNORECASE)


def _fields_message(expected: set[str], actual: set[Any]) -> str:
    missing = sorted(expected - actual)
    extra = sorted((actual - expected), key=lambda value: str(value))
    return f"missing={missing}, extra={extra}"


def _object_with_exact_fields(
    parent: Mapping[str, Any], name: str, expected: set[str]
) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"report section {name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(f"invalid {name} fields ({_fields_message(expected, actual)})")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{path} is out of range")
    return value


def _number(value: Any, path: str, *, exclusive_minimum: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite")
    if converted < 0 or (exclusive_minimum and converted <= 0):
        raise ValueError(f"{path} is out of range")
    return converted


def _string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "a non-empty string" if nonempty else "a string"
        raise ValueError(f"{path} must be {qualifier}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _torch_versions() -> tuple[str, str]:
    try:
        # The controller must not import PyTorch or initialize CUDA.  A healthy
        # worker replaces the CUDA value with the version it actually used.
        return importlib.metadata.version("torch"), "unavailable"
    except importlib.metadata.PackageNotFoundError:
        return "unavailable", "unavailable"


def _device_snapshot(device: int, include_identifiers: bool) -> dict[str, Any]:
    # Failure/timeout reports are constructed by the controller, which must not initialize CUDA.
    # A healthy worker replaces this conservative snapshot with its observed device facts.
    return {
        "ordinal": device,
        "name": "redacted",
        "total_vram_bytes": 0,
        "compute_capability": "unavailable",
        "identifiers_included": include_identifiers,
    }


def report_schema_version(configuration: Mapping[str, Any]) -> int:
    """Select the frozen v1 report for off mode and v2 for compression.

    Missing ``compression`` is intentionally equivalent to ``off`` so every
    Phase 4b caller retains its exact report/protocol contract.
    """

    mode = configuration.get("compression", "off")
    return (
        SCHEMA_VERSION
        if isinstance(mode, str) and mode.lower() == "off"
        else SCHEMA_VERSION_COMPRESSION
    )


def trace_schema_version(configuration: Mapping[str, Any]) -> int:
    return report_schema_version(configuration)


def _empty_compression_section(configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": str(configuration.get("compression", "adaptive")),
        "codec": str(configuration.get("compression_codec", "auto")),
        "host_store_cap_bytes": int(configuration.get("host_store_cap_bytes", 0)),
        "host_headroom_bytes": int(configuration.get("host_headroom_bytes", 0)),
        "host_budget_bytes": 0,
        "host_budget_peak_bytes": 0,
        "conversion_scratch_peak_bytes": 0,
        "logical_bytes": 0,
        "stored_bytes": 0,
        "stored_peak_bytes": 0,
        "raw_bytes": 0,
        "compressed_bytes": 0,
        "implicit_zero_bytes": 0,
        "invalid_bytes": 0,
        "logical_h2d_bytes": 0,
        "pcie_h2d_bytes": 0,
        "pcie_h2d_payload_bytes": 0,
        "pcie_h2d_metadata_bytes": 0,
        "logical_d2h_bytes": 0,
        "pcie_d2h_bytes": 0,
        "pcie_d2h_payload_bytes": 0,
        "pcie_d2h_metadata_bytes": 0,
        "rejected_candidate_logical_d2h_bytes": 0,
        "hot_allocation_d2h_bytes": 0,
        "non_hot_allocation_d2h_bytes": 0,
        "raw_path_decisions": 0,
        "cpu_lz4_gpu_decode_decisions": 0,
        "gpu_lz4_decisions": 0,
        "never_compress_decisions": 0,
        "raw_fallbacks": 0,
        "cpu_codec_fallbacks": 0,
        "gpu_codec_fallbacks": 0,
        "compression_attempts": 0,
        "compression_commits": 0,
        "decompression_attempts": 0,
        "decompression_commits": 0,
        "generations_created": 0,
        "generations_committed": 0,
        "generations_discarded": 0,
        "workspace_bytes": 0,
        "workspace_peak_bytes": 0,
        "slot_bytes": 0,
        "slot_peak_bytes": 0,
        "spill_reserved_bytes": 0,
        "spill_reserved_peak_bytes": 0,
        "cpu_encode_nanoseconds": 0,
        "cpu_decode_nanoseconds": 0,
        "gpu_encode_nanoseconds": 0,
        "gpu_decode_nanoseconds": 0,
        "verification_nanoseconds": 0,
        "codec_events_recorded": 0,
        "codec_events_retired": 0,
    }


def empty_report(
    configuration: Mapping[str, Any],
    *,
    exit_code: int = EXIT_RUNTIME,
    status: str = "failed",
    message: str = "worker did not produce a final report",
    include_identifiers: bool = False,
) -> dict[str, Any]:
    torch_version, cuda_version = _torch_versions()
    device = int(configuration.get("device", 0))
    schema_version = report_schema_version(configuration)
    report = {
        "schema_version": schema_version,
        "report_type": REPORT_TYPE,
        "build": {
            "version": "0.1.0-dev",
            "git_commit": os.environ.get("XVRAM_GIT_COMMIT", "unknown"),
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "torch_target_version": "2.11",
            "bridge_api_version": schema_version,
        },
        "system": {
            "os": platform.system() or "unknown",
            "architecture": platform.machine() or "unknown",
            "python_version": platform.python_version(),
        },
        "device": _device_snapshot(device, include_identifiers),
        "configuration": dict(configuration),
        "model": {
            "kind": str(configuration.get("model", "llama2-like")),
            "layers": int(configuration.get("layers", 0)),
            "batch": int(configuration.get("batch", 1)),
            "sequence": int(configuration.get("sequence", 32)),
            "hidden": int(configuration.get("hidden", 4096)),
            "intermediate": int(configuration.get("intermediate", 11008)),
            "heads": int(configuration.get("heads", 32)),
            "logical_bytes": 0,
            "parameter_bytes": 0,
            "activation_bytes": 0,
        },
        "graph": {
            "capture": "torch.export.strict",
            "hash": "",
            "backend_hash": "",
            "allowlist_version": ALLOWLIST_VERSION,
            "allowlist_hash": ALLOWLIST_HASH,
            "node_count": 0,
            "region_count": 0,
            "unsupported_nodes": [],
        },
        "execution": {
            "leases_acquired": 0,
            "leases_sealed": 0,
            "leases_retired": 0,
            "tiled_gemm_regions": 0,
            "tiled_gemm_tiles": 0,
            "events_recorded": 0,
            "events_retired": 0,
            "live_views_peak": 0,
            "live_views_final": 0,
            "regions_completed": 0,
            "region_timings_ms": [],
            "scratch_peak_bytes": 0,
            "scratch_cap_bytes": int(configuration.get("scratch_cap_bytes", 0)),
            "prefetch_distance": int(configuration.get("prefetch_distance", 2)),
            "trace_records": 0,
            "trace_complete": False,
        },
        "cache": {
            "target_bytes": int(configuration.get("cache_target_bytes", 0)),
            "chunk_bytes": int(configuration.get("chunk_size_bytes", 0)),
            "resident_bytes": 0,
            "resident_peak_bytes": 0,
            "maps": 0,
            "set_access": 0,
            "unmaps": 0,
            "hits": 0,
            "misses": 0,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "weight_d2h_bytes": 0,
            "evictions": 0,
            "frame_reuses": 0,
            "prefetch_submitted": 0,
            "prefetch_retired": 0,
            "unsafe_remaps": 0,
            "unsafe_transitions": 0,
        },
        "verification": {
            "reference_digest": "",
            "output_digest": "",
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mismatch_count": 0,
        },
        "proof": {
            "logical_exceeds_vram": False,
            "real_oversubscription": False,
            "stable_addresses": False,
            "maps_match_set_access": False,
            "event_safe": False,
            "zero_weight_writeback": False,
            "bounded_scratch": False,
            "reference_matches": False,
            "zero_live_views": False,
            "full_cleanup": False,
        },
        "outcome": {"status": status, "exit_code": exit_code, "message": message},
        "cleanup": {
            "worker_reaped": False,
            "leases_drained": False,
            "views_released": False,
            "mappings_unmapped": False,
            "handles_released": False,
            "reservations_released": False,
            "streams_destroyed": False,
            "context_released": False,
            "complete": False,
        },
        "diagnostics": [],
    }
    if schema_version == SCHEMA_VERSION_COMPRESSION:
        report["compression"] = _empty_compression_section(configuration)
    return report


def finalize_proof(report: dict[str, Any]) -> dict[str, Any]:
    """Populate proof flags from counters without relaxing success semantics."""

    model = report["model"]
    device = report["device"]
    execution = report["execution"]
    cache = report["cache"]
    verification = report["verification"]
    cleanup = report["cleanup"]
    logical = int(model["logical_bytes"])
    vram = int(device["total_vram_bytes"])
    scratch_peak = int(execution["scratch_peak_bytes"])
    scratch_cap = int(execution["scratch_cap_bytes"])
    proof = report["proof"]
    proof.update(
        {
            "logical_exceeds_vram": vram > 0 and logical > vram,
            "real_oversubscription": (
                vram > 0
                and logical > vram
                and int(cache["evictions"]) > 0
                and int(cache["frame_reuses"]) > 0
            ),
            "maps_match_set_access": int(cache["maps"]) == int(cache["set_access"]),
            "event_safe": (
                int(cache["unsafe_remaps"]) == 0
                and int(cache["unsafe_transitions"]) == 0
                and int(execution["events_recorded"]) == int(execution["events_retired"])
            ),
            "zero_weight_writeback": int(cache["weight_d2h_bytes"]) == 0,
            "bounded_scratch": scratch_cap >= 0 and scratch_peak <= scratch_cap,
            "reference_matches": (
                int(verification["mismatch_count"]) == 0
                and bool(verification["reference_digest"])
                and verification["reference_digest"] == verification["output_digest"]
            ),
            "zero_live_views": int(execution["live_views_final"]) == 0,
            "full_cleanup": bool(cleanup["complete"]),
        }
    )
    return report


def success_semantics(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    try:
        validate_report_envelope(report)
    except (TypeError, ValueError) as exc:
        return False, [f"invalid report envelope: {exc}"]

    errors: list[str] = []
    outcome = report["outcome"]
    if outcome.get("exit_code") != EXIT_COMPLETED or outcome.get("status") != "completed":
        errors.append("outcome is not completed")
    proof = report["proof"]
    required = (
        "stable_addresses",
        "maps_match_set_access",
        "event_safe",
        "zero_weight_writeback",
        "bounded_scratch",
        "reference_matches",
        "zero_live_views",
        "full_cleanup",
    )
    model = report["model"]
    device = report["device"]
    if model["logical_bytes"] > device["total_vram_bytes"] > 0:
        required = ("logical_exceeds_vram", "real_oversubscription", *required)
    errors.extend(f"proof.{name} is false" for name in required if not proof.get(name, False))

    graph = report["graph"]
    hashes = (graph["hash"], graph["backend_hash"])
    if (
        graph["hash"] != graph["backend_hash"]
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
    ):
        errors.append("graph hash and backend hash are not equal and non-empty")
    if graph["allowlist_version"] != ALLOWLIST_VERSION or graph["allowlist_hash"] != ALLOWLIST_HASH:
        errors.append("graph allowlist contract is not canonical")
    if graph["unsupported_nodes"]:
        errors.append("graph contains unsupported nodes")
    if graph["node_count"] <= 0:
        errors.append("graph node count is zero")

    execution = report["execution"]
    regions = graph["region_count"]
    if regions <= 0 or execution["regions_completed"] != regions:
        errors.append("graph regions were not completely reconciled")
    if len(execution["region_timings_ms"]) != execution["regions_completed"]:
        errors.append("region timings do not reconcile with completed regions")
    leases = (
        execution["leases_acquired"],
        execution["leases_sealed"],
        execution["leases_retired"],
    )
    tiled_regions = execution["tiled_gemm_regions"]
    tiled_tiles = execution["tiled_gemm_tiles"]
    if leases[0] != leases[1] or leases[1] != leases[2]:
        errors.append("leases are not fully reconciled")
    if tiled_regions < 0 or tiled_tiles < tiled_regions:
        errors.append("tiled GEMM regions and tiles do not reconcile")
    if leases[2] + tiled_regions != regions:
        errors.append("lease and tiled GEMM regions do not reconcile")
    events = (execution["events_recorded"], execution["events_retired"])
    minimum_event_boundaries = leases[2] + tiled_tiles
    if int(report.get("schema_version", 0)) == SCHEMA_VERSION_COMPRESSION:
        minimum_event_boundaries += report["compression"]["codec_events_retired"]
    if (
        events[0] <= 0
        or events[0] != events[1]
        or events[1] < minimum_event_boundaries
    ):
        errors.append("events are zero or do not reconcile with execution boundaries")
    if (
        (regions > tiled_regions and execution["live_views_peak"] <= 0)
        or execution["live_views_final"] != 0
    ):
        errors.append("managed tensor views are zero or not fully released")
    cache = report["cache"]
    if cache["target_bytes"] <= 0 or cache["chunk_bytes"] <= 0:
        errors.append("cache target or chunk size is zero")
    if cache["resident_bytes"] != 0 or cache["resident_peak_bytes"] > cache["target_bytes"]:
        errors.append("cache residency does not reconcile with its live target")
    if cache["maps"] != cache["unmaps"]:
        errors.append("cache mappings were not fully unmapped")
    if cache["maps"] <= 0 or cache["misses"] <= 0 or cache["h2d_bytes"] <= 0:
        errors.append("cache did not perform observable managed transfers")
    if cache["weight_d2h_bytes"] > cache["d2h_bytes"]:
        errors.append("weight D2H bytes exceed total D2H bytes")
    if cache["prefetch_submitted"] != cache["prefetch_retired"]:
        errors.append("prefetch requests were not fully retired")
    for name in ("reference_digest", "output_digest"):
        digest = report["verification"][name]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            errors.append(f"verification.{name} is not lowercase SHA-256")
    configured_identifiers = bool(report["configuration"].get("include_identifiers", False))
    if bool(report["device"]["identifiers_included"]) != configured_identifiers:
        errors.append("device identifier privacy flag does not match configuration")
    if not execution["trace_complete"]:
        errors.append("execution trace is incomplete")
    if report["device"]["total_vram_bytes"] <= 0:
        errors.append("device VRAM was not observed")
    parameter_bytes = report["model"]["parameter_bytes"]
    activation_bytes = report["model"]["activation_bytes"]
    logical_bytes = report["model"]["logical_bytes"]
    if parameter_bytes <= 0 or activation_bytes <= 0 or logical_bytes < parameter_bytes + activation_bytes:
        errors.append("model byte accounting is empty or inconsistent")
    if not configured_identifiers and report["device"]["name"] != "redacted":
        errors.append("device name was not redacted")
    if report.get("diagnostics"):
        errors.append("diagnostics are not empty")
    if int(report.get("schema_version", 0)) == SCHEMA_VERSION_COMPRESSION:
        compression = report["compression"]
        decisions = (
            compression["raw_path_decisions"]
            + compression["cpu_lz4_gpu_decode_decisions"]
            + compression["gpu_lz4_decisions"]
            + compression["never_compress_decisions"]
        )
        if compression["policy"] not in {"adaptive", "capacity"}:
            errors.append("compression policy is invalid")
        if compression["logical_bytes"] != logical_bytes:
            errors.append("compression logical bytes do not reconcile with the model")
        if compression["logical_h2d_bytes"] != report["cache"]["h2d_bytes"]:
            errors.append("compression logical H2D bytes do not reconcile with the cache")
        if compression["logical_d2h_bytes"] != (
            report["cache"]["d2h_bytes"]
            + compression["rejected_candidate_logical_d2h_bytes"]
        ):
            errors.append("compression logical D2H bytes do not reconcile with the cache")
        if compression["hot_allocation_d2h_bytes"] != report["cache"]["weight_d2h_bytes"]:
            errors.append("measured weight D2H bytes do not reconcile with the cache")
        if (
            compression["hot_allocation_d2h_bytes"]
            + compression["non_hot_allocation_d2h_bytes"]
            != report["cache"]["d2h_bytes"]
        ):
            errors.append("per-allocation-class D2H bytes do not reconcile")
        if (
            compression["stored_bytes"]
            != compression["raw_bytes"] + compression["compressed_bytes"]
        ):
            errors.append("compression host representations do not reconcile")
        if compression["stored_bytes"] > compression["stored_peak_bytes"]:
            errors.append("compression stored bytes exceed the observed peak")
        if compression["pcie_h2d_bytes"] != (
            compression["pcie_h2d_payload_bytes"]
            + compression["pcie_h2d_metadata_bytes"]
        ):
            errors.append("H2D PCIe bytes do not reconcile with payload and metadata")
        if compression["pcie_d2h_bytes"] != (
            compression["pcie_d2h_payload_bytes"]
            + compression["pcie_d2h_metadata_bytes"]
        ):
            errors.append("D2H PCIe bytes do not reconcile with payload and metadata")
        if compression["pcie_h2d_payload_bytes"] > compression["logical_h2d_bytes"]:
            errors.append("H2D PCIe payload exceeds logical H2D bytes")
        if compression["pcie_d2h_payload_bytes"] > compression["logical_d2h_bytes"]:
            errors.append("D2H PCIe payload exceeds logical D2H bytes")
        if (
            compression["rejected_candidate_logical_d2h_bytes"]
            > compression["logical_d2h_bytes"]
        ):
            errors.append("rejected-candidate logical D2H exceeds all logical D2H attempts")
        if decisions <= 0:
            errors.append("compression path decisions are empty")
        if compression["compression_commits"] > compression["compression_attempts"]:
            errors.append("compression commits exceed attempts")
        if compression["decompression_commits"] > compression["decompression_attempts"]:
            errors.append("decompression commits exceed attempts")
        if compression["generations_created"] != (
            compression["generations_committed"]
            + compression["generations_discarded"]
        ):
            errors.append("compression generations do not reconcile")
        host_cap = compression["host_store_cap_bytes"]
        if host_cap <= 0:
            errors.append("effective compression host storage cap was not observed")
        elif compression["host_budget_peak_bytes"] > host_cap:
            errors.append("compression host budget exceeds its effective cap")
        if compression["host_headroom_bytes"] <= 0:
            errors.append("effective compression host headroom was not observed")
        if compression["host_budget_bytes"] > compression["host_budget_peak_bytes"]:
            errors.append("compression current host budget exceeds its observed peak")
        if compression["stored_bytes"] > compression["host_budget_bytes"]:
            errors.append("compression backing bytes exceed the charged host budget")
        if compression["stored_peak_bytes"] > compression["host_budget_peak_bytes"]:
            errors.append("compression backing peak exceeds the charged host budget peak")
        if (
            compression["conversion_scratch_peak_bytes"]
            > compression["host_budget_peak_bytes"]
        ):
            errors.append("compression conversion scratch exceeds the host budget peak")
        if compression["codec_events_recorded"] != compression["codec_events_retired"]:
            errors.append("compression codec events do not reconcile")
        if (
            compression["workspace_peak_bytes"]
            > report["configuration"]["compression_scratch_cap_bytes"]
        ):
            errors.append("compression workspace exceeds its configured cap")
    return not errors, errors


def validate_report_envelope(report: Mapping[str, Any]) -> None:
    """Reject protocol data that cannot be a v1 or v2 report.

    Full JSON Schema validation remains part of contract/acceptance tests.  The
    controller also performs this dependency-free boundary check so an invalid
    or privacy-unsafe worker payload is never written as if it were a report.
    """

    try:
        _validate_report_envelope(report)
    except ValueError:
        raise
    except Exception as exc:
        # Protocol payloads are untrusted.  Any unexpected container behavior or
        # conversion problem must remain a validation failure, never a controller crash.
        raise ValueError(f"report validation failed safely: {exc}") from exc


def _validate_report_envelope(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("report must be an object")
    schema_version = report.get("schema_version")
    if schema_version not in {SCHEMA_VERSION, SCHEMA_VERSION_COMPRESSION}:
        raise ValueError("invalid report type or schema version")
    expected_top = (
        _TOP_LEVEL_KEYS
        if schema_version == SCHEMA_VERSION
        else _TOP_LEVEL_KEYS_V2
    )
    actual_top = set(report)
    if actual_top != expected_top:
        raise ValueError(
            f"invalid report sections ({_fields_message(expected_top, actual_top)})"
        )
    if report.get("report_type") != REPORT_TYPE:
        raise ValueError("invalid report type or schema version")

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if (
                    lowered
                    in {
                        "cuda_va",
                        "raw_va",
                        "stream_handle",
                        "native_stream_handle",
                        "pointer",
                        "raw_pointer",
                        "native_pointer",
                        "device_pointer",
                    }
                    or "virtual_address" in lowered
                    or lowered.endswith("_pointer")
                ):
                    raise ValueError(f"report contains forbidden field {key}")
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
        elif isinstance(value, str):
            tagged_identity = _FORBIDDEN_IDENTITY_TEXT.search(value)
            bare_address = path != ("configuration", "seed") and _BARE_ADDRESS_TEXT.search(
                value
            )
            if tagged_identity or bare_address:
                raise ValueError(
                    "report contains a serialized CUDA address or stream handle"
                )

    visit(report)

    sections = {
        name: _object_with_exact_fields(report, name, fields)
        for name, fields in _SECTION_KEYS.items()
    }

    build = sections["build"]
    for name in ("version", "git_commit", "torch_version", "cuda_version"):
        _string(build[name], f"build.{name}", nonempty=True)
    if (
        build["torch_target_version"] != "2.11"
        or build["bridge_api_version"] != schema_version
    ):
        raise ValueError("invalid Stable-ABI build contract")

    system = sections["system"]
    for name in _SECTION_KEYS["system"]:
        _string(system[name], f"system.{name}", nonempty=True)

    device = sections["device"]
    _integer(device["ordinal"], "device.ordinal")
    _string(device["name"], "device.name", nonempty=True)
    _integer(device["total_vram_bytes"], "device.total_vram_bytes")
    _string(device["compute_capability"], "device.compute_capability", nonempty=True)
    _boolean(device["identifiers_included"], "device.identifiers_included")

    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("report section configuration must be an object")
    actual_configuration = set(configuration)
    required_configuration = set(_CONFIGURATION_REQUIRED_KEYS)
    if schema_version == SCHEMA_VERSION_COMPRESSION:
        required_configuration |= _CONFIGURATION_V2_KEYS
    allowed_configuration = required_configuration | _CONFIGURATION_OPTIONAL_KEYS
    if (
        not required_configuration.issubset(actual_configuration)
        or not actual_configuration.issubset(allowed_configuration)
    ):
        missing = sorted(required_configuration - actual_configuration)
        extra = sorted(
            actual_configuration - allowed_configuration, key=lambda value: str(value)
        )
        raise ValueError(
            f"invalid configuration fields (missing={missing}, extra={extra})"
        )
    _integer(configuration["device"], "configuration.device")
    if configuration["model"] not in {"llama2-like", "operator-smoke"}:
        raise ValueError("configuration.model is invalid")
    _number(configuration["model_ratio"], "configuration.model_ratio", exclusive_minimum=True)
    for name in ("layers", "batch", "sequence", "hidden", "intermediate", "heads"):
        _integer(configuration[name], f"configuration.{name}", minimum=1)
    if configuration["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("configuration.dtype is invalid")
    if configuration["policy"] not in {"clock", "lru"}:
        raise ValueError("configuration.policy is invalid")
    for name in ("cache_target_bytes", "device_headroom_bytes", "scratch_cap_bytes"):
        _integer(configuration[name], f"configuration.{name}")
    _integer(configuration["chunk_size_bytes"], "configuration.chunk_size_bytes", minimum=1)
    _integer(configuration["prefetch_distance"], "configuration.prefetch_distance", maximum=8)
    if configuration["sdpa_backend"] not in {"math", "flash_attention"}:
        raise ValueError("configuration.sdpa_backend is invalid")
    seed = _string(configuration["seed"], "configuration.seed", nonempty=True)
    if not seed.startswith("0x") or not 1 <= len(seed[2:]) <= 16:
        raise ValueError("configuration.seed is invalid")
    try:
        int(seed[2:], 16)
    except ValueError as exc:
        raise ValueError("configuration.seed is invalid") from exc
    if "include_identifiers" in configuration:
        _boolean(configuration["include_identifiers"], "configuration.include_identifiers")
    if schema_version == SCHEMA_VERSION_COMPRESSION:
        if configuration["compression"] not in {"adaptive", "capacity"}:
            raise ValueError("configuration.compression is invalid")
        if configuration["compression_codec"] not in {"auto", "lz4"}:
            raise ValueError("configuration.compression_codec is invalid")
        if configuration["state_pattern"] not in {"incompressible", "structured"}:
            raise ValueError("configuration.state_pattern is invalid")
        for name in (
            "host_store_cap_bytes",
            "host_headroom_bytes",
            "compression_scratch_cap_bytes",
        ):
            _integer(configuration[name], f"configuration.{name}")
        _integer(configuration["compression_scratch_cap_bytes"], "configuration.compression_scratch_cap_bytes", minimum=1)
        _integer(configuration["codec_slots"], "configuration.codec_slots", minimum=2, maximum=8)
        _integer(configuration["codec_workers"], "configuration.codec_workers", minimum=1, maximum=8)

    model = sections["model"]
    _string(model["kind"], "model.kind", nonempty=True)
    _integer(model["layers"], "model.layers")
    for name in ("batch", "sequence", "hidden", "intermediate", "heads"):
        _integer(model[name], f"model.{name}", minimum=1)
    for name in ("logical_bytes", "parameter_bytes", "activation_bytes"):
        _integer(model[name], f"model.{name}")

    graph = sections["graph"]
    if graph["capture"] != "torch.export.strict":
        raise ValueError("graph.capture is invalid")
    _string(graph["hash"], "graph.hash")
    _string(graph["backend_hash"], "graph.backend_hash")
    if graph["allowlist_version"] != ALLOWLIST_VERSION:
        raise ValueError("graph.allowlist_version is invalid")
    allowlist_hash = _string(graph["allowlist_hash"], "graph.allowlist_hash")
    if len(allowlist_hash) != 64 or any(character not in "0123456789abcdef" for character in allowlist_hash):
        raise ValueError("graph.allowlist_hash must be lowercase SHA-256")
    _integer(graph["node_count"], "graph.node_count")
    _integer(graph["region_count"], "graph.region_count")
    unsupported = graph["unsupported_nodes"]
    if not isinstance(unsupported, list) or not all(isinstance(item, str) for item in unsupported):
        raise ValueError("graph.unsupported_nodes must be an array of strings")

    execution = sections["execution"]
    execution_counts = (
        "leases_acquired",
        "leases_sealed",
        "leases_retired",
        "tiled_gemm_regions",
        "tiled_gemm_tiles",
        "events_recorded",
        "events_retired",
        "live_views_peak",
        "live_views_final",
        "regions_completed",
        "scratch_peak_bytes",
        "scratch_cap_bytes",
        "trace_records",
    )
    for name in execution_counts:
        _integer(execution[name], f"execution.{name}")
    _integer(execution["prefetch_distance"], "execution.prefetch_distance", maximum=8)
    timings = execution["region_timings_ms"]
    if not isinstance(timings, list):
        raise ValueError("execution.region_timings_ms must be an array")
    for index, value in enumerate(timings):
        _number(value, f"execution.region_timings_ms[{index}]")
    _boolean(execution["trace_complete"], "execution.trace_complete")

    cache = sections["cache"]
    for name in _SECTION_KEYS["cache"]:
        _integer(cache[name], f"cache.{name}")

    verification = sections["verification"]
    _string(verification["reference_digest"], "verification.reference_digest")
    _string(verification["output_digest"], "verification.output_digest")
    _number(verification["max_abs_error"], "verification.max_abs_error")
    _number(verification["max_rel_error"], "verification.max_rel_error")
    _integer(verification["mismatch_count"], "verification.mismatch_count")

    proof = sections["proof"]
    for name in _SECTION_KEYS["proof"]:
        _boolean(proof[name], f"proof.{name}")

    outcome = sections["outcome"]
    if outcome["status"] not in {"completed", "skipped", "corruption", "oom", "timeout", "failed"}:
        raise ValueError("outcome.status is invalid")
    if outcome["exit_code"] not in _ALLOWED_EXIT_CODES or isinstance(outcome["exit_code"], bool):
        raise ValueError("outcome.exit_code is invalid")
    _string(outcome["message"], "outcome.message")

    cleanup = sections["cleanup"]
    for name in _SECTION_KEYS["cleanup"]:
        _boolean(cleanup[name], f"cleanup.{name}")

    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError("report diagnostics must be an array")
    for index, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, Mapping) or set(diagnostic) != _DIAGNOSTIC_KEYS:
            raise ValueError(f"diagnostics[{index}] has invalid fields")
        _string(diagnostic["stage"], f"diagnostics[{index}].stage", nonempty=True)
        _string(diagnostic["code"], f"diagnostics[{index}].code", nonempty=True)
        _string(diagnostic["message"], f"diagnostics[{index}].message")

    if schema_version == SCHEMA_VERSION_COMPRESSION:
        compression = _object_with_exact_fields(
            report, "compression", _COMPRESSION_SECTION_KEYS
        )
        if compression["policy"] not in {"adaptive", "capacity"}:
            raise ValueError("compression.policy is invalid")
        if compression["codec"] not in {"auto", "lz4"}:
            raise ValueError("compression.codec is invalid")
        for name in _COMPRESSION_SECTION_KEYS - {"policy", "codec"}:
            _integer(compression[name], f"compression.{name}")


def validate_trace_record(record: Mapping[str, Any]) -> None:
    schema_version = record.get("schema_version")
    expected_keys = (
        _TRACE_KEYS
        if schema_version == SCHEMA_VERSION
        else _TRACE_KEYS_V2
        if schema_version == SCHEMA_VERSION_COMPRESSION
        else set()
    )
    if set(record) != expected_keys:
        raise ValueError("invalid trace record fields")
    if record.get("record_type") != "xvram.pytorch_trace":
        raise ValueError("invalid trace record contract")
    for name in ("sequence", "monotonic_ns", "bytes"):
        value = record.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"trace {name} must be a non-negative integer")
    if int(record["sequence"]) < 1:
        raise ValueError("trace sequence must be positive")
    for name in ("region_id", "allocation_id"):
        value = record.get(name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"trace {name} must be null or a positive integer")
    trace_kinds = _TRACE_KINDS if schema_version == 1 else _TRACE_KINDS_V2
    if record.get("kind") not in trace_kinds:
        raise ValueError("trace kind is invalid")
    if not isinstance(record.get("operation"), str) or not record["operation"]:
        raise ValueError("trace operation must be non-empty")
    if not isinstance(record.get("reason"), str):
        raise ValueError("trace reason must be a string")
    if any(
        pattern.search(value)
        for value in (record["operation"], record["reason"])
        for pattern in (_FORBIDDEN_IDENTITY_TEXT, _BARE_ADDRESS_TEXT)
    ):
        raise ValueError("trace contains a serialized CUDA address or stream handle")
    if schema_version == SCHEMA_VERSION_COMPRESSION:
        for name in (
            "chunk_index",
            "source_generation",
            "target_generation",
            "slot_generation",
        ):
            value = record.get(name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"trace {name} must be null or non-negative")
        if record.get("representation") not in {
            None,
            "invalid",
            "implicit_zero",
            "raw",
            "lz4_blocks",
        }:
            raise ValueError("trace representation is invalid")
        if record.get("codec_path") not in {
            None,
            "raw",
            "cpu_lz4_gpu_decode",
            "nvcomp_gpu_codec",
        }:
            raise ValueError("trace codec path is invalid")


def dumps(report: Mapping[str, Any], *, compact: bool = False) -> str:
    if compact:
        return json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
