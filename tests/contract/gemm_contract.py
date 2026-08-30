#!/usr/bin/env python3
"""GPU-independent contract checks for the Phase 3 tiled GEMM report."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


MIB = 1024 * 1024
GIB = 1024 * MIB


def timing(samples: int = 4) -> dict[str, Any]:
    if samples == 0:
        return {
            "sample_count": 0,
            "total_ms": None,
            "minimum_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "maximum_ms": None,
        }
    return {
        "sample_count": samples,
        "total_ms": 20.0,
        "minimum_ms": 4.0,
        "median_ms": 5.0,
        "p95_ms": 6.0,
        "maximum_ms": 6.0,
    }


def completed_report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": "xvram.gemm_bench",
        "generated_at_utc": "2026-08-30T12:00:00Z",
        "build": {
            "version": "0.1.0-dev",
            "git_commit": "contract-fixture",
            "compiler": "contract",
            "build_type": "Debug",
            "cuda_headers_version": 13030,
        },
        "system": {
            "os_name": "Windows",
            "os_version": "11",
            "architecture": "x86_64",
            "logical_processor_count": 16,
            "physical_memory_bytes": 32 * GIB,
            "available_memory_bytes": 20 * GIB,
        },
        "device": {
            "ordinal": 0,
            "name": "NVIDIA GeForce RTX 3070",
            "uuid": None,
            "luid": None,
            "pci_bus_id": None,
            "driver_model": "wddm",
            "total_memory_bytes": 8 * GIB,
            "free_memory_bytes_start": 7 * GIB,
            "free_memory_bytes_end": 7 * GIB,
            "safe_device_budget_bytes": 6 * GIB,
            "wddm_budget_bytes_start": 8 * GIB,
            "wddm_usage_bytes_start": 1 * GIB,
            "wddm_available_bytes_start": 7 * GIB,
            "wddm_available_bytes_minimum": 6 * GIB,
            "wddm_available_bytes_end": 7 * GIB,
            "vmm_supported": True,
            "uva_supported": True,
            "minimum_granularity_bytes": 64 * 1024,
            "recommended_granularity_bytes": 2 * MIB,
            "compute_capability_major": 8,
            "compute_capability_minor": 6,
        },
        "configuration": {
            "requested_device_ordinal": 0,
            "requested_m": 65_536,
            "requested_n": 32_768,
            "requested_k": 2_048,
            "scenario": "oversubscribed",
            "a_data_type": "fp32",
            "b_data_type": "fp32",
            "c_data_type": "fp32",
            "compute_mode": "tf32",
            "a_layout": "row_major",
            "b_layout": "column_major",
            "c_layout": "row_major",
            "a_operation": "n",
            "b_operation": "t",
            "alpha": 1.0,
            "beta": 0.0,
            "requested_chunk_bytes": 64 * MIB,
            "effective_chunk_bytes": 64 * MIB,
            "requested_cache_target_bytes": None,
            "initial_cache_target_bytes": 6 * GIB,
            "requested_workspace_cap_bytes": 4 * MIB,
            "effective_workspace_cap_bytes": 4 * MIB,
            "context_mode": "isolated",
            "policy": "clock",
            "staging_slots": 4,
            "prefetch_distance": 2,
            "passes": 1,
            "budget_poll_ms": 100,
            "stall_timeout_ms": 5_000,
            "timeout_ms": 300_000,
            "max_tile_ms": 250,
            "identifiers_included": False,
        },
        "plans": [{
            "plan_id": 1,
            "status": "completed",
            "m": 65_536,
            "n": 32_768,
            "k": 2_048,
            "a_data_type": "fp32",
            "b_data_type": "fp32",
            "c_data_type": "fp32",
            "effective_compute_mode": "tf32",
            "tile_m": 4_096,
            "tile_n": 4_096,
            "tile_k": 1_024,
            "tile_count": 256,
            "maximum_working_set_bytes": 512 * MIB,
            "workspace_bytes": 4 * MIB,
            "uses_cublas_lt": True,
            "heuristic_algorithm_count": 8,
            "replan_count": 0,
        }],
        "workloads": [{
            "name": "oversubscribed",
            "plan_id": 1,
            "status": "completed",
            "passes_completed": 1,
            "tiles_total": 256,
            "tiles_retired": 256,
            "logical_allocation_bytes": 10 * GIB,
            "oversubscribed": True,
            "cache_hits": 128,
            "cache_misses": 384,
            "cache_hit_rate": 0.25,
            "bytes_h2d": 20 * GIB,
            "bytes_d2h": 8 * GIB,
            "clean_evictions": 64,
            "dirty_evictions": 128,
            "writebacks_completed": 128,
            "mapping_count": 512,
            "unmap_count": 512,
            "set_access_count": 512,
            "handle_reuse_count": 384,
            "event_boundary_count": 512,
            "unsafe_remap_count": 0,
            "unsafe_transition_count": 0,
            "elapsed_ms": 1_000.0,
            "achieved_tflops": 8.79,
            "output_digest128": "a" * 32,
        }],
        "numerics": {
            "validation_mode": "structured",
            "reference_precision": "analytic",
            "absolute_tolerance": 0.001,
            "relative_tolerance": 0.001,
            "maximum_absolute_error": 0.0001,
            "maximum_relative_error": 0.0001,
            "elements_verified": 65_536 * 32_768,
            "mismatch_count": 0,
            "first_mismatch_byte_offset": None,
            "expected_digest128": "a" * 32,
            "output_digest128": "a" * 32,
            "all_results_match": True,
        },
        "cache": {
            "target_bytes_initial": 6 * GIB,
            "target_bytes_minimum": 5 * GIB,
            "target_bytes_maximum": 6 * GIB,
            "target_bytes_end": 6 * GIB,
            "resident_bytes_peak": 5 * GIB,
            "pinned_staging_bytes": 256 * MIB,
            "workspace_bytes_peak": 4 * MIB,
            "physical_frame_count_peak": 80,
            "physical_handle_create_count": 128,
            "physical_handle_release_count": 128,
            "handle_reuse_count": 384,
            "mapping_count": 512,
            "unmap_count": 512,
            "set_access_count": 512,
            "event_boundary_count": 512,
            "unsafe_remap_count": 0,
            "unsafe_transition_count": 0,
            "cache_hits": 128,
            "cache_misses": 384,
            "cache_hit_rate": 0.25,
            "clean_evictions": 64,
            "dirty_evictions": 128,
            "writebacks_completed": 128,
            "target_shrink_count": 1,
            "target_grow_count": 1,
            "target_oom_retry_count": 0,
        },
        "telemetry": {
            "cuda_driver_version": 13030,
            "cublas_version": 130501,
            "cublas_lt_version": 130501,
            "cublas_lt_available": True,
            "cublas_library_source": "system",
            "algorithms_selected": 1,
            "algorithm_cache_hits": 255,
            "budget_sample_count": 20,
            "cuda_free_bytes_minimum": 1 * GIB,
            "cuda_free_bytes_end": 7 * GIB,
            "wddm_available_bytes_minimum": 6 * GIB,
            "wddm_available_bytes_end": 7 * GIB,
            "replans": 0,
            "watchdog_rejections": 0,
            "total_elapsed_ms": 1_100.0,
            "plan_timing": timing(1),
            "tile_timing": timing(256),
            "gemm_timing": timing(256),
            "h2d_timing": timing(384),
            "d2h_timing": timing(128),
            "writeback_timing": timing(128),
        },
        "proof": {
            "logical_allocation_bytes": 10 * GIB,
            "total_vram_bytes": 8 * GIB,
            "maximum_cache_target_bytes": 6 * GIB,
            "maximum_working_set_bytes": 512 * MIB,
            "workspace_cap_bytes": 4 * MIB,
            "logical_data_exceeds_vram": True,
            "cache_smaller_than_logical": True,
            "tiled_execution_verified": True,
            "handles_reused": True,
            "stable_virtual_addresses_verified": True,
            "set_access_after_map_verified": True,
            "event_boundaries_verified": True,
            "no_physical_aliases_verified": True,
            "dirty_writeback_verified": True,
            "cache_target_respected": True,
            "workspace_bounded": True,
            "all_workloads_match_reference": True,
            "raw_virtual_addresses_omitted": True,
        },
        "outcome": {
            "status": "completed",
            "reason": None,
            "exit_code": 0,
            "stage": None,
            "operation": None,
            "native_code": None,
            "native_name": None,
            "message": None,
            "plan_id": None,
            "tile_sequence": None,
            "logical_byte_offset": None,
        },
        "cleanup": {
            "complete": True,
            "operations_drained": True,
            "events_drained": True,
            "events_destroyed": True,
            "streams_destroyed": True,
            "cublas_handles_destroyed": True,
            "mappings_removed": True,
            "physical_handles_released": True,
            "workspace_released": True,
            "virtual_reservations_released": True,
            "pinned_staging_released": True,
            "host_backing_released": True,
            "context_released": True,
            "worker_terminated": True,
        },
        "diagnostics": [],
    }


def partial_report(template: dict[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(template)
    report["device"] = None
    report["plans"] = []
    report["workloads"] = []
    for key in report["cache"]:
        report["cache"][key] = None
    for key, value in report["telemetry"].items():
        report["telemetry"][key] = timing(0) if isinstance(value, dict) else None
    for key in report["proof"]:
        report["proof"][key] = None
    for key in report["numerics"]:
        if key == "validation_mode":
            report["numerics"][key] = "none"
        elif key == "reference_precision":
            report["numerics"][key] = "none"
        else:
            report["numerics"][key] = None
    report["outcome"].update(
        status="skipped", reason="device_unavailable", exit_code=23, stage="planning"
    )
    for key in report["cleanup"]:
        report["cleanup"][key] = None
    report["cleanup"]["worker_terminated"] = True
    return report


def errors(validator: Draft202012Validator, instance: dict[str, Any]) -> list[Any]:
    return sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))


def assert_no_raw_virtual_addresses(value: Any) -> None:
    forbidden = {
        "base_address",
        "device_address",
        "device_pointer",
        "native_stream",
        "raw_va",
        "virtual_address",
        "virtual_address_base",
        "workspace_device_address",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            assert_no_raw_virtual_addresses(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_raw_virtual_addresses(child)


def assert_completed_semantics(report: dict[str, Any]) -> None:
    device = report["device"]
    proof = report["proof"]
    cache = report["cache"]
    workload = report["workloads"][0]
    plan = report["plans"][0]
    assert device is not None
    assert workload["plan_id"] == plan["plan_id"]
    assert workload["tiles_retired"] == workload["tiles_total"] == plan["tile_count"]
    assert proof["logical_allocation_bytes"] > device["total_memory_bytes"]
    assert proof["maximum_cache_target_bytes"] < proof["logical_allocation_bytes"]
    assert plan["maximum_working_set_bytes"] <= proof["maximum_cache_target_bytes"]
    assert plan["workspace_bytes"] <= proof["workspace_cap_bytes"]
    assert cache["resident_bytes_peak"] <= cache["target_bytes_maximum"]
    assert cache["workspace_bytes_peak"] <= proof["workspace_cap_bytes"]
    assert cache["mapping_count"] == cache["set_access_count"] == cache["unmap_count"]
    assert cache["dirty_evictions"] == cache["writebacks_completed"]
    assert cache["unsafe_remap_count"] == 0
    assert cache["unsafe_transition_count"] == 0
    assert workload["cache_hits"] + workload["cache_misses"] > 0
    assert workload["output_digest128"] == report["numerics"]["output_digest128"]
    assert report["numerics"]["expected_digest128"] == report["numerics"]["output_digest128"]
    assert_no_raw_virtual_addresses(report)


def main() -> int:
    schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    completed = completed_report()
    assert not errors(validator, completed)
    assert_completed_semantics(completed)

    small = copy.deepcopy(completed)
    small["workloads"][0]["logical_allocation_bytes"] = 1 * GIB
    small["workloads"][0]["oversubscribed"] = False
    small["proof"]["logical_allocation_bytes"] = 1 * GIB
    small["proof"]["logical_data_exceeds_vram"] = False
    small["proof"]["cache_smaller_than_logical"] = False
    small["proof"]["handles_reused"] = False
    assert not errors(validator, small)

    skipped = partial_report(completed)
    assert not errors(validator, skipped)

    for reason, code, stage in (
        ("unsupported_type", 23, "planning"),
        ("data_mismatch", 24, "verification"),
        ("device_oom", 25, "allocation"),
        ("budget_pressure", 25, "budget"),
        ("timeout", 26, "watchdog"),
        ("cublas_error", 27, "gemm"),
        ("protocol_error", 27, "protocol"),
        ("cleanup_error", 27, "cleanup"),
        ("output_io_error", 74, "output"),
    ):
        failure = partial_report(completed)
        failure["outcome"].update(
            status="skipped" if code == 23 else ("timed_out" if code == 26 else "failed"),
            reason=reason,
            exit_code=code,
            stage=stage,
            operation="contract_fixture",
            message=f"synthetic {reason}",
        )
        assert not errors(validator, failure), reason

    identifiers = copy.deepcopy(completed)
    identifiers["configuration"]["identifiers_included"] = True
    identifiers["device"]["uuid"] = "GPU-00112233-4455-6677-8899-aabbccddeeff"
    identifiers["device"]["luid"] = "0102030405060708"
    identifiers["device"]["pci_bus_id"] = "00000000:65:00.0"
    assert not errors(validator, identifiers)

    invalid_cases = []
    for section, key, value in (
        ("proof", "tiled_execution_verified", False),
        ("proof", "raw_virtual_addresses_omitted", False),
        ("proof", "all_workloads_match_reference", False),
        ("cache", "unsafe_remap_count", 1),
        ("cleanup", "complete", False),
        ("numerics", "all_results_match", False),
        ("numerics", "mismatch_count", 1),
    ):
        invalid = copy.deepcopy(completed)
        invalid[section][key] = value
        invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["raw_va"] = 0x1234
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["configuration"]["identifiers_included"] = False
    invalid["device"]["uuid"] = "GPU-private"
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(skipped)
    invalid["outcome"]["reason"] = None
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["plans"][0]["status"] = "failed"
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["workloads"][0]["status"] = "failed"
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["numerics"]["expected_digest128"] = None
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["cache"]["mapping_count"] = None
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(skipped)
    invalid["outcome"]["exit_code"] = 27
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(skipped)
    invalid["outcome"].update(status="timed_out", exit_code=27)
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(skipped)
    invalid["outcome"].update(status="failed", exit_code=23)
    invalid_cases.append(invalid)
    assert all(errors(validator, invalid) for invalid in invalid_cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
