#!/usr/bin/env python3
"""Contract fixtures and semantic checks for adaptive compression report v1."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema


GIB = 1024**3
FORBIDDEN_KEY_PARTS = ("virtual_address", "stream_handle", "device_pointer", "host_pointer")
FORBIDDEN_KEYS = {"cuda_va", "raw_va", "cuda_address", "stream"}
SAFE_PROOF_KEYS = {"stable_virtual_addresses_verified", "raw_virtual_addresses_omitted"}
POINTER_VALUE = re.compile(r"^0x[0-9a-fA-F]{8,}$")


def timing(samples: int = 0, value: float | None = None) -> dict[str, object]:
    return {
        "sample_count": samples,
        "total_ms": value,
        "minimum_ms": value,
        "median_ms": value,
        "p95_ms": value,
        "maximum_ms": value,
    }


def configuration() -> dict[str, object]:
    return {
        "requested_device_ordinal": 0,
        "requested_logical_bytes": 24 * GIB,
        "effective_logical_bytes": 24 * GIB,
        "requested_chunk_bytes": 64 * 1024**2,
        "effective_chunk_bytes": 64 * 1024**2,
        "requested_cache_target_bytes": 7 * GIB,
        "initial_cache_target_bytes": 7 * GIB,
        "compression_policy": "adaptive",
        "path": "all",
        "codec": "lz4",
        "replacement_policy": "clock",
        "scenario": "suite",
        "passes": 2,
        "warmup_passes": 2,
        "measurement_passes": 5,
        "host_store_cap_bytes": 16 * GIB,
        "host_headroom_bytes": 4 * GIB,
        "device_headroom_bytes": 512 * 1024**2,
        "compression_scratch_cap_bytes": 256 * 1024**2,
        "codec_slots": 2,
        "codec_workers": 2,
        "staging_slots": 4,
        "prefetch_distance": 2,
        "budget_poll_ms": 100,
        "stall_timeout_ms": 5000,
        "timeout_ms": 900000,
        "seed_hex": "585652414d503035",
        "sizing_mode": "explicit",
        "trace_enabled": True,
        "identifiers_included": False,
    }


def empty_report(status: str, exit_code: int, reason: str | None) -> dict[str, Any]:
    optional_backing = {
        name: None
        for name in (
            "logical_bytes",
            "host_store_cap_bytes",
            "host_bytes_current",
            "host_bytes_peak",
            "raw_bytes_current",
            "raw_bytes_peak",
            "compressed_bytes_current",
            "compressed_bytes_peak",
            "spill_reserved_bytes_peak",
            "conversion_scratch_bytes_peak",
            "invalid_chunks",
            "implicit_zero_chunks",
            "raw_chunks",
            "lz4_chunks",
            "generations_created",
            "generations_committed",
            "generations_discarded",
            "atomic_commit_failures",
            "expansion_rejections",
            "effective_stored_ratio",
        )
    }
    optional_codec = {
        name: None
        for name in (
            "cpu_codec_version",
            "nvcomp_available",
            "nvcomp_version",
            "nvcomp_library_source",
            "nvcomp_library_sha256",
            "cpu_encode_operations",
            "cpu_decode_operations",
            "gpu_encode_operations",
            "gpu_decode_operations",
            "raw_path_decisions",
            "cpu_lz4_gpu_decisions",
            "gpu_lz4_decisions",
            "never_compress_decisions",
            "calibration_samples",
            "fallback_count",
            "verification_failures",
            "codec_slots_peak",
            "workspace_bytes_peak",
        )
    }
    optional_telemetry = {
        name: None
        for name in (
            "total_elapsed_ms",
            "logical_h2d_bytes",
            "pcie_h2d_bytes",
            "logical_d2h_bytes",
            "pcie_d2h_bytes",
            "mapping_count",
            "set_access_count",
            "unmap_count",
            "event_record_count",
            "event_retire_count",
            "handle_reuse_count",
            "unsafe_remap_count",
            "unsafe_transition_count",
            "writeback_count",
            "target_shrink_count",
            "target_grow_count",
            "budget_sample_count",
            "cache_target_bytes_minimum",
            "cache_target_bytes_maximum",
            "trace_records_emitted",
            "trace_records_dropped",
            "trace_complete",
        )
    }
    proof_names = (
        "logical_data_exceeds_vram",
        "cache_smaller_than_logical",
        "authoritative_compressed_backing_verified",
        "no_expansion_stored",
        "generation_atomicity_verified",
        "write_admission_verified",
        "gpu_encode_before_d2h_verified",
        "compressed_h2d_reduction_verified",
        "stable_virtual_addresses_verified",
        "maps_match_set_access",
        "event_boundaries_verified",
        "host_budget_respected",
        "device_budget_respected",
        "all_workloads_match_reference",
        "path_digests_match",
        "policy_digests_match",
        "raw_virtual_addresses_omitted",
    )
    cleanup_names = (
        "complete",
        "operations_drained",
        "codec_slots_drained",
        "events_drained",
        "events_destroyed",
        "streams_destroyed",
        "mappings_removed",
        "physical_handles_released",
        "codec_workspace_released",
        "virtual_reservations_released",
        "pinned_staging_released",
        "spill_reservations_released",
        "host_backing_released",
        "context_released",
        "trace_closed",
        "worker_terminated",
    )
    return {
        "schema_version": 1,
        "report_type": "xvram.adaptive_compression",
        "generated_at_utc": "2026-09-03T00:00:00Z",
        "build": {
            "version": "0.1.0-dev",
            "git_commit": "fixture",
            "compiler": "fixture",
            "build_type": "Release",
            "cuda_headers_version": 13030,
        },
        "system": {
            "os_name": "Fixture OS",
            "os_version": "1.0",
            "architecture": "x86_64",
            "logical_processor_count": 16,
            "physical_memory_bytes": 64 * GIB,
            "available_memory_bytes": 48 * GIB,
        },
        "device": None,
        "configuration": configuration(),
        "workloads": [],
        "backing": optional_backing,
        "codec": {
            "container_version": "xvram_lz4_blocks_v1",
            "block_bytes": 65536,
            "cpu_codec": "lz4",
            **optional_codec,
            "cpu_encode_timing": timing(),
            "cpu_decode_timing": timing(),
            "gpu_encode_timing": timing(),
            "gpu_decode_timing": timing(),
        },
        "telemetry": {
            **optional_telemetry,
            "raw_transfer_timing": timing(),
            "compressed_transfer_timing": timing(),
            "writeback_timing": timing(),
        },
        "proof": {name: None for name in proof_names},
        "outcome": {
            "status": status,
            "reason": reason,
            "exit_code": exit_code,
            "stage": None,
            "operation": None,
            "native_code": None,
            "native_name": None,
            "message": "fixture",
            "scenario": None,
            "allocation_id": None,
            "chunk_index": None,
            "generation": None,
            "logical_byte_offset": None,
        },
        "cleanup": {name: None for name in cleanup_names},
        "diagnostics": [],
    }


def completed_report() -> dict[str, Any]:
    report = empty_report("completed", 0, None)
    report["device"] = {
        "ordinal": 0,
        "name": "Fixture RTX 3070",
        "uuid": None,
        "luid": None,
        "pci_bus_id": None,
        "driver_model": "wddm",
        "total_memory_bytes": 8 * GIB,
        "free_memory_bytes_start": 7 * GIB,
        "free_memory_bytes_end": 7 * GIB,
        "safe_device_budget_bytes": 7 * GIB,
        "wddm_budget_bytes_start": 8 * GIB,
        "wddm_usage_bytes_start": GIB,
        "wddm_available_bytes_minimum": 6 * GIB,
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "vmm_supported": True,
        "uva_supported": True,
    }
    report["workloads"] = [
        {
            "scenario": "compressible-read",
            "compression_policy": "adaptive",
            "path": "cpu_lz4_gpu_decode",
            "replacement_policy": "clock",
            "status": "completed",
            "logical_bytes": 24 * GIB,
            "logical_chunk_count": 384,
            "operations_retired": 384,
            "warmup_passes_completed": 2,
            "measurement_passes_completed": 5,
            "raw_input_bytes": 24 * GIB,
            "stored_bytes": 6 * GIB,
            "stored_ratio": 0.25,
            "logical_h2d_bytes": 24 * GIB,
            "pcie_h2d_bytes": 6 * GIB,
            "logical_d2h_bytes": 12 * GIB,
            "pcie_d2h_bytes": 3 * GIB,
            "raw_path_decisions": 0,
            "cpu_lz4_gpu_decisions": 256,
            "gpu_lz4_decisions": 128,
            "never_compress_decisions": 0,
            "fallback_count": 0,
            "mappings": 384,
            "set_access": 384,
            "unmaps": 384,
            "events_recorded": 768,
            "events_retired": 768,
            "unsafe_remaps": 0,
            "unsafe_transitions": 0,
            "elapsed_ms": 1000.0,
            "expected_digest128": "a" * 32,
            "output_digest128": "a" * 32,
            "mismatch_count": 0,
            "first_mismatch_byte_offset": None,
        }
    ]
    report["backing"].update(
        {
            "logical_bytes": 24 * GIB,
            "host_store_cap_bytes": 16 * GIB,
            "host_bytes_current": 6 * GIB,
            "host_bytes_peak": 8 * GIB,
            "raw_bytes_current": 0,
            "raw_bytes_peak": 2 * GIB,
            "compressed_bytes_current": 6 * GIB,
            "compressed_bytes_peak": 6 * GIB,
            "spill_reserved_bytes_peak": GIB,
            "conversion_scratch_bytes_peak": 128 * 1024**2,
            "invalid_chunks": 0,
            "implicit_zero_chunks": 0,
            "raw_chunks": 0,
            "lz4_chunks": 384,
            "generations_created": 512,
            "generations_committed": 500,
            "generations_discarded": 12,
            "atomic_commit_failures": 0,
            "expansion_rejections": 0,
            "effective_stored_ratio": 0.25,
        }
    )
    report["codec"].update(
        {
            "cpu_codec_version": "1.10.0",
            "nvcomp_available": True,
            "nvcomp_version": "5.3.0.16",
            "nvcomp_library_source": "app_local",
            "nvcomp_library_sha256": "b" * 64,
            "cpu_encode_operations": 256,
            "cpu_decode_operations": 0,
            "gpu_encode_operations": 128,
            "gpu_decode_operations": 384,
            "raw_path_decisions": 0,
            "cpu_lz4_gpu_decisions": 256,
            "gpu_lz4_decisions": 128,
            "never_compress_decisions": 0,
            "calibration_samples": 4,
            "fallback_count": 0,
            "verification_failures": 0,
            "codec_slots_peak": 2,
            "workspace_bytes_peak": 128 * 1024**2,
            "cpu_encode_timing": timing(256, 4.0),
            "gpu_encode_timing": timing(128, 2.0),
            "gpu_decode_timing": timing(384, 1.0),
        }
    )
    report["telemetry"].update(
        {
            "total_elapsed_ms": 1000.0,
            "logical_h2d_bytes": 24 * GIB,
            "pcie_h2d_bytes": 6 * GIB,
            "logical_d2h_bytes": 12 * GIB,
            "pcie_d2h_bytes": 3 * GIB,
            "mapping_count": 384,
            "set_access_count": 384,
            "unmap_count": 384,
            "event_record_count": 768,
            "event_retire_count": 768,
            "handle_reuse_count": 256,
            "unsafe_remap_count": 0,
            "unsafe_transition_count": 0,
            "writeback_count": 128,
            "target_shrink_count": 1,
            "target_grow_count": 1,
            "budget_sample_count": 10,
            "cache_target_bytes_minimum": 6 * GIB,
            "cache_target_bytes_maximum": 7 * GIB,
            "trace_records_emitted": 1024,
            "trace_records_dropped": 0,
            "trace_complete": True,
            "raw_transfer_timing": timing(2, 8.0),
            "compressed_transfer_timing": timing(5, 5.0),
            "writeback_timing": timing(128, 3.0),
        }
    )
    for key in report["proof"]:
        report["proof"][key] = True
    for key in report["cleanup"]:
        report["cleanup"][key] = True
    return report


def semantic_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report["outcome"]["status"] != "completed" or report["outcome"]["exit_code"] != 0:
        errors.append("outcome")
    workloads = report["workloads"]
    for workload in workloads:
        if not (workload["mappings"] == workload["set_access"] == workload["unmaps"]):
            errors.append("workload mapping accounting")
        if workload["events_recorded"] != workload["events_retired"]:
            errors.append("workload event accounting")
        decisions = (
            workload["raw_path_decisions"]
            + workload["cpu_lz4_gpu_decisions"]
            + workload["gpu_lz4_decisions"]
        )
        if decisions != workload["operations_retired"]:
            errors.append("workload decision accounting")
        if workload["pcie_h2d_bytes"] > workload["logical_h2d_bytes"]:
            errors.append("H2D byte accounting")
        if workload["pcie_d2h_bytes"] > workload["logical_d2h_bytes"]:
            errors.append("D2H byte accounting")
        if workload["expected_digest128"] != workload["output_digest128"]:
            errors.append("digest mismatch")
    telemetry = report["telemetry"]
    if not (
        telemetry["mapping_count"]
        == telemetry["set_access_count"]
        == telemetry["unmap_count"]
    ):
        errors.append("global mapping accounting")
    if telemetry["event_record_count"] != telemetry["event_retire_count"]:
        errors.append("global event accounting")
    backing = report["backing"]
    if backing["generations_created"] != (
        backing["generations_committed"] + backing["generations_discarded"]
    ):
        errors.append("generation accounting")
    if backing["host_bytes_current"] != (
        backing["raw_bytes_current"] + backing["compressed_bytes_current"]
    ):
        errors.append("host byte accounting")
    if backing["host_bytes_peak"] > backing["host_store_cap_bytes"]:
        errors.append("host budget")
    if report["codec"]["workspace_bytes_peak"] > report["configuration"]["compression_scratch_cap_bytes"]:
        errors.append("codec workspace budget")
    if telemetry["trace_records_dropped"] != 0 or not telemetry["trace_complete"]:
        errors.append("trace accounting")
    return errors


def contains_private_runtime_value(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = key.lower()
            if lower not in SAFE_PROOF_KEYS and (
                lower in FORBIDDEN_KEYS or any(part in lower for part in FORBIDDEN_KEY_PARTS)
            ):
                return True
            if contains_private_runtime_value(child):
                return True
    elif isinstance(value, list):
        return any(contains_private_runtime_value(child) for child in value)
    elif isinstance(value, str) and POINTER_VALUE.fullmatch(value):
        return True
    return False


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compression_contract.py <report-schema> <trace-schema>")
        return 64
    report_schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    trace_schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    validate_report = jsonschema.Draft202012Validator(report_schema).validate
    validate_trace = jsonschema.Draft202012Validator(trace_schema).validate

    completed = completed_report()
    validate_report(completed)
    if errors := semantic_errors(completed):
        raise AssertionError(errors)
    if contains_private_runtime_value(completed):
        raise AssertionError("report exposes a CUDA address, pointer, or stream handle")

    for code, status, reason in (
        (23, "skipped", "prerequisite"),
        (24, "corruption", "reference_mismatch"),
        (25, "oom", "host_budget"),
        (26, "timeout", "worker_timeout"),
        (27, "failed", "cuda_failure"),
        (74, "failed", "output_io"),
    ):
        validate_report(empty_report(status, code, reason))

    trace = {
        "schema_version": 1,
        "report_type": "xvram.compression_trace",
        "sequence": 1,
        "monotonic_time_ns": 123,
        "event": "backing_transition",
        "allocation_id": 7,
        "chunk_index": 11,
        "operation_id": 19,
        "source_generation": 3,
        "target_generation": 4,
        "slot_generation": None,
        "from_representation": "raw",
        "to_representation": "lz4_blocks",
        "path": "cpu_lz4_gpu_decode",
        "logical_bytes": 64 * 1024**2,
        "physical_bytes": 16 * 1024**2,
        "reason": "adaptive_cost_win",
        "speculative": False,
    }
    validate_trace(trace)
    if contains_private_runtime_value(trace):
        raise AssertionError("trace exposes a CUDA address, pointer, or stream handle")

    invalid = copy.deepcopy(completed)
    invalid["telemetry"]["mapping_count"] += 1
    if not semantic_errors(invalid):
        raise AssertionError("semantic accounting did not reject mismatched maps")

    invalid = copy.deepcopy(completed)
    invalid["cuda_va"] = "0x123456789abcdef0"
    if jsonschema.Draft202012Validator(report_schema).is_valid(invalid):
        raise AssertionError("strict schema accepted a raw CUDA VA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
