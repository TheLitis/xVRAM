#!/usr/bin/env python3
"""GPU-independent contracts for Phase 2 residency reports and trace records."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


MIB = 1024 * 1024
GIB = 1024 * MIB


def timing(samples: int = 2) -> dict[str, Any]:
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
        "total_ms": 2.0,
        "minimum_ms": 0.5,
        "median_ms": 1.0,
        "p95_ms": 1.5,
        "maximum_ms": 1.5,
    }


def workload(policy: str) -> dict[str, Any]:
    return {
        "scenario": "sequential",
        "policy": policy,
        "status": "completed",
        "operations_retired": 100,
        "passes_completed": 2,
        "logical_bytes": 12 * GIB,
        "logical_chunk_count": 192,
        "maximum_working_set_bytes": 2 * 64 * MIB,
        "read_operations": 20,
        "read_write_operations": 60,
        "write_only_operations": 20,
        "cache_hits": 80,
        "cache_misses": 20,
        "cache_hit_rate": 0.8,
        "bytes_h2d": 1_000,
        "bytes_d2h": 600,
        "clean_evictions": 5,
        "dirty_evictions": 6,
        "writebacks_completed": 6,
        "prefetch_issued": 10,
        "prefetch_useful": 7,
        "prefetch_wasted": 2,
        "prefetch_cancelled": 1,
        "prefetch_promoted": 2,
        "sequential_bypasses": 3,
        "target_shrink_count": 1,
        "target_grow_count": 1,
        "mapping_count": 20,
        "unmap_count": 20,
        "set_access_count": 20,
        "handle_reuse_count": 10,
        "event_boundary_count": 20,
        "unsafe_remap_count": 0,
        "unsafe_transition_count": 0,
        "stable_addresses_verified": True,
        "full_verification_completed": True,
        "matches_cpu": True,
        "elapsed_ms": 50.0,
        "throughput_gib_per_second": 4.0,
        "expected_digest128": "a" * 32,
        "output_digest128": "a" * 32,
        "mismatch_count": 0,
        "first_mismatch_byte_offset": None,
    }


AGGREGATE_COUNTERS = (
    "handle_reuse_count",
    "mapping_count",
    "unmap_count",
    "set_access_count",
    "event_boundary_count",
    "unsafe_remap_count",
    "unsafe_transition_count",
    "cache_hits",
    "cache_misses",
    "clean_evictions",
    "dirty_evictions",
    "writebacks_completed",
    "prefetch_issued",
    "prefetch_useful",
    "prefetch_wasted",
    "prefetch_cancelled",
    "prefetch_promoted",
    "sequential_bypasses",
    "target_shrink_count",
    "target_grow_count",
)


def completed_report() -> dict[str, Any]:
    workloads = [workload("clock"), workload("lru")]
    cache = {
        "target_bytes_initial": 6 * GIB,
        "target_bytes_minimum": 5 * GIB,
        "target_bytes_maximum": 6 * GIB,
        "target_bytes_end": 6 * GIB,
        "resident_bytes_peak": 6 * GIB,
        "pinned_staging_bytes": 256 * MIB,
        "physical_frame_count_peak": 96,
        "physical_handle_create_count": 192,
        "physical_handle_release_count": 192,
        "target_oom_retry_count": 0,
        "maximum_working_set_bytes": 2 * 64 * MIB,
    }
    for key in AGGREGATE_COUNTERS:
        cache[key] = sum(item[key] for item in workloads)
    cache["cache_hit_rate"] = cache["cache_hits"] / (
        cache["cache_hits"] + cache["cache_misses"]
    )

    return {
        "schema_version": 1,
        "report_type": "xvram.residency_cache",
        "generated_at_utc": "2026-08-30T12:00:00Z",
        "build": {
            "version": "0.1.0-dev",
            "git_commit": "contract-fixture",
            "compiler": "contract",
            "build_type": "Debug",
            "cuda_headers_version": 13000,
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
        },
        "configuration": {
            "requested_device_ordinal": 0,
            "requested_logical_bytes": 12 * GIB,
            "effective_logical_bytes": 12 * GIB,
            "requested_chunk_bytes": 64 * MIB,
            "effective_chunk_bytes": 64 * MIB,
            "requested_cache_target_bytes": None,
            "initial_cache_target_bytes": 6 * GIB,
            "staging_slots": 4,
            "policy": "both",
            "prefetch_distance": 2,
            "scenario": "sequential",
            "passes": 2,
            "pressure_bytes": None,
            "host_headroom_bytes": 8 * GIB,
            "device_headroom_bytes": 512 * MIB,
            "budget_poll_ms": 100,
            "stall_timeout_ms": 5_000,
            "timeout_ms": 300_000,
            "seed_hex": "585652414d503032",
            "sizing_mode": "mixed",
            "trace_enabled": True,
            "identifiers_included": False,
        },
        "workloads": workloads,
        "cache": cache,
        "telemetry": {
            "total_elapsed_ms": 100.0,
            "bytes_h2d": sum(item["bytes_h2d"] for item in workloads),
            "bytes_d2h": sum(item["bytes_d2h"] for item in workloads),
            "budget_sample_count": 20,
            "cuda_free_bytes_minimum": 1 * GIB,
            "cuda_free_bytes_end": 7 * GIB,
            "wddm_available_bytes_minimum": 6 * GIB,
            "wddm_available_bytes_end": 7 * GIB,
            "resident_occupancy_peak": 1.0,
            "resident_occupancy_mean": 0.75,
            "remap_timing": timing(),
            "h2d_timing": timing(),
            "kernel_timing": timing(),
            "d2h_timing": timing(),
            "writeback_timing": timing(),
            "trace_records_emitted": 500,
            "trace_records_dropped": 0,
            "trace_complete": True,
        },
        "proof": {
            "logical_bytes": 12 * GIB,
            "chunk_bytes": 64 * MIB,
            "logical_chunk_count": 192,
            "maximum_cache_target_bytes": 6 * GIB,
            "pinned_staging_bytes": 256 * MIB,
            "kernel_module_version": 1,
            "kernel_module_sha256": "b" * 64,
            "pattern_version": "xvram_residency_workload_v1",
            "cpu_reference_digest128": "a" * 32,
            "logical_data_exceeds_vram": True,
            "cache_smaller_than_logical": True,
            "handles_reused": True,
            "stable_virtual_addresses_verified": True,
            "set_access_after_map_verified": True,
            "event_boundaries_verified": True,
            "no_physical_aliases_verified": True,
            "dirty_writeback_verified": True,
            "staging_pool_bounded": True,
            "cache_target_respected": True,
            "all_workloads_match_cpu": True,
            "policies_match": True,
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
            "policy": None,
            "scenario": None,
            "allocation_id": None,
            "chunk_index": None,
            "logical_byte_offset": None,
        },
        "cleanup": {
            "complete": True,
            "transactions_drained": True,
            "prefetch_drained": True,
            "writebacks_completed": True,
            "events_drained": True,
            "events_destroyed": True,
            "streams_destroyed": True,
            "module_unloaded": True,
            "mappings_removed": True,
            "physical_handles_released": True,
            "device_allocations_released": True,
            "virtual_reservations_released": True,
            "pinned_staging_released": True,
            "host_backing_released": True,
            "context_destroyed": True,
            "trace_closed": True,
            "worker_terminated": True,
        },
        "diagnostics": [],
    }


def partial_report(template: dict[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(template)
    report["device"] = None
    report["workloads"] = []
    for key in report["cache"]:
        report["cache"][key] = None
    for key, value in report["telemetry"].items():
        report["telemetry"][key] = timing(0) if isinstance(value, dict) else None
    for key in report["proof"]:
        if key == "kernel_module_version":
            report["proof"][key] = 0
        elif key in ("kernel_module_sha256", "pattern_version"):
            report["proof"][key] = ""
        else:
            report["proof"][key] = None
    report["outcome"].update(
        status="skipped", reason="device_unavailable", exit_code=23, stage="planning"
    )
    for key in report["cleanup"]:
        report["cleanup"][key] = None
    report["cleanup"]["worker_terminated"] = True
    return report


def failure_report(
    template: dict[str, Any], reason: str, exit_code: int, stage: str
) -> dict[str, Any]:
    report = partial_report(template)
    report["outcome"].update(
        status="failed",
        reason=reason,
        exit_code=exit_code,
        stage=stage,
        operation="contract_fixture",
        message=f"synthetic {reason} fixture",
    )
    return report


def trace_record(event: str = "state_transition") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": "xvram.residency_trace",
        "sequence": 1,
        "monotonic_time_ns": 100,
        "event": event,
        "allocation_id": 1 if event == "state_transition" else None,
        "chunk_index": 2 if event == "state_transition" else None,
        "operation_id": None,
        "transaction_id": None,
        "from_state": "host_clean" if event == "state_transition" else None,
        "to_state": "mapping" if event == "state_transition" else None,
        "bytes": 64 * MIB,
        "reason": "demand",
        "policy": "clock",
        "speculative": False,
        "generation": 1 if event == "state_transition" else None,
    }


def errors(validator: Draft202012Validator, instance: dict[str, Any]) -> list[Any]:
    return sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path))


def assert_no_raw_virtual_addresses(value: Any) -> None:
    forbidden = {
        "base_address",
        "device_pointer",
        "raw_va",
        "virtual_address",
        "virtual_address_base",
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
    cache = report["cache"]
    proof = report["proof"]
    workloads = report["workloads"]
    assert device is not None
    assert proof["logical_bytes"] > device["total_memory_bytes"]
    assert cache["resident_bytes_peak"] <= cache["target_bytes_maximum"]
    assert cache["target_bytes_maximum"] < proof["logical_bytes"]
    assert cache["pinned_staging_bytes"] == (
        report["configuration"]["effective_chunk_bytes"]
        * report["configuration"]["staging_slots"]
    )
    assert {item["policy"] for item in workloads} == {"clock", "lru"}
    for key in AGGREGATE_COUNTERS:
        assert cache[key] == sum(item[key] for item in workloads), key
    assert cache["mapping_count"] == cache["set_access_count"]
    assert cache["mapping_count"] == cache["unmap_count"]
    assert cache["dirty_evictions"] == cache["writebacks_completed"]
    assert cache["unsafe_remap_count"] == 0
    assert cache["unsafe_transition_count"] == 0
    for item in workloads:
        assert item["operations_retired"] == (
            item["read_operations"]
            + item["read_write_operations"]
            + item["write_only_operations"]
        )
        assert item["prefetch_issued"] == (
            item["prefetch_useful"]
            + item["prefetch_wasted"]
            + item["prefetch_cancelled"]
        )
        assert item["expected_digest128"] == item["output_digest128"]
        assert item["output_digest128"] == proof["cpu_reference_digest128"]
    assert_no_raw_virtual_addresses(report)


def main() -> int:
    report_schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    trace_schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(report_schema)
    Draft202012Validator.check_schema(trace_schema)
    report_validator = Draft202012Validator(report_schema, format_checker=FormatChecker())
    trace_validator = Draft202012Validator(trace_schema)

    completed = completed_report()
    assert not errors(report_validator, completed)
    assert_completed_semantics(completed)

    sub_vram = copy.deepcopy(completed)
    sub_vram["proof"]["logical_data_exceeds_vram"] = False
    assert not errors(report_validator, sub_vram)

    skipped = partial_report(completed)
    assert not errors(report_validator, skipped)
    assert skipped["cleanup"]["complete"] is None

    failure_fixtures = (
        failure_report(completed, "working_set_too_large", 23, "planning"),
        failure_report(completed, "data_mismatch", 24, "verification"),
        failure_report(completed, "host_oom", 25, "host_backing"),
        failure_report(completed, "device_oom", 25, "physical_allocation"),
        failure_report(completed, "budget_pressure", 25, "budget"),
        failure_report(completed, "timeout", 26, "watchdog"),
        failure_report(completed, "cuda_error", 27, "kernel"),
        failure_report(completed, "platform_error", 27, "protocol"),
        failure_report(completed, "cleanup_error", 27, "cleanup"),
    )
    for fixture in failure_fixtures:
        assert not errors(report_validator, fixture), fixture["outcome"]["reason"]

    incomplete_trace = copy.deepcopy(completed)
    incomplete_trace["outcome"].update(
        status="failed",
        reason="trace_io_error",
        exit_code=74,
        stage="trace",
        operation="write_trace",
        message="synthetic incomplete trace fixture",
    )
    incomplete_trace["telemetry"]["trace_records_dropped"] = 1
    incomplete_trace["telemetry"]["trace_complete"] = False
    incomplete_trace["cleanup"]["complete"] = False
    incomplete_trace["cleanup"]["trace_closed"] = False
    assert not errors(report_validator, incomplete_trace)

    identifiers = copy.deepcopy(completed)
    identifiers["configuration"]["identifiers_included"] = True
    identifiers["device"]["uuid"] = "GPU-00112233-4455-6677-8899-aabbccddeeff"
    identifiers["device"]["luid"] = "0102030405060708"
    identifiers["device"]["pci_bus_id"] = "00000000:65:00.0"
    assert not errors(report_validator, identifiers)

    invalid_cases = []
    for section, key, value in (
        ("proof", "handles_reused", False),
        ("cache", "unsafe_remap_count", 1),
        ("cleanup", "complete", False),
    ):
        invalid = copy.deepcopy(completed)
        invalid[section][key] = value
        invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["workloads"][0]["matches_cpu"] = False
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["proof"]["policies_match"] = False
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["device"]["uuid"] = "GPU-private"
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(completed)
    invalid["raw_va"] = 0x1234
    invalid_cases.append(invalid)
    invalid = copy.deepcopy(skipped)
    invalid["outcome"]["reason"] = None
    invalid_cases.append(invalid)
    assert all(errors(report_validator, invalid) for invalid in invalid_cases)

    transition = trace_record()
    assert not errors(trace_validator, transition)
    target = trace_record("target_changed")
    assert not errors(trace_validator, target)
    operation = trace_record("operation_retired")
    operation["operation_id"] = 3
    operation["transaction_id"] = 4
    assert not errors(trace_validator, operation)

    for emitted_event in (
        "allocation",
        "kernel_submit",
        "kernel_retire",
        "prefetch_useful",
        "victim_select",
        "victim_selected",
        "writeback_prepare",
        "pressure_acquire",
        "pressure_release",
    ):
        assert not errors(trace_validator, trace_record(emitted_event)), emitted_event

    invalid_trace = copy.deepcopy(transition)
    invalid_trace["raw_va"] = 0x1234
    assert errors(trace_validator, invalid_trace)
    invalid_trace = copy.deepcopy(transition)
    invalid_trace["to_state"] = None
    assert errors(trace_validator, invalid_trace)
    invalid_trace = copy.deepcopy(transition)
    invalid_trace["event"] = "unknown"
    assert errors(trace_validator, invalid_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
