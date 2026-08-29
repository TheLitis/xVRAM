#!/usr/bin/env python3
"""GPU- and CLI-independent contract tests for the Phase 1 VMM POC report."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def run_fixture(executable: str, mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        [executable, mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


def validation_errors(
    validator: Draft202012Validator, report: dict[str, Any]
) -> list[Any]:
    return sorted(validator.iter_errors(report), key=lambda item: list(item.absolute_path))


def assert_no_raw_virtual_addresses(value: Any) -> None:
    forbidden = {
        "base_address",
        "device_pointer",
        "raw_va",
        "virtual_address",
        "virtual_address_base",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value), forbidden.intersection(value)
        for child in value.values():
            assert_no_raw_virtual_addresses(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_raw_virtual_addresses(child)


def not_run_mode(template: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(template)
    for key in result:
        if key == "status":
            result[key] = "not_run"
        elif key == "remap_timing":
            result[key] = {
                "sample_count": 0,
                "total_ms": None,
                "minimum_ms": None,
                "median_ms": None,
                "p95_ms": None,
                "maximum_ms": None,
            }
        else:
            result[key] = None
    return result


def assert_completed_semantics(report: dict[str, Any]) -> None:
    configuration = report["configuration"]
    device = report["device"]
    modes = report["modes"]
    proof = report["proof"]
    pipeline = modes["pipeline"]
    reference = modes["reference"]

    assert report["outcome"]["status"] == "completed"
    assert report["outcome"]["exit_code"] == 0
    assert device is not None
    assert proof["effective_logical_bytes"] > device["total_memory_bytes"]
    assert configuration["effective_window_slots"] == 2
    assert proof["resident_physical_bytes"] == (
        proof["effective_chunk_bytes"] * configuration["effective_window_slots"]
    )
    assert proof["pinned_staging_bytes"] == proof["resident_physical_bytes"]
    assert proof["resident_physical_bytes"] < proof["effective_logical_bytes"]
    assert pipeline["physical_handle_count"] == configuration["effective_window_slots"]
    assert proof["tile_visit_count"] == proof["logical_chunk_count"] * configuration["passes"]
    assert proof["address_revisit_count"] == proof["logical_chunk_count"] * (
        configuration["passes"] - 1
    )

    expected_elements = proof["logical_element_count"] * configuration["passes"]
    for mode in (reference, pipeline):
        assert mode["status"] == "completed"
        assert mode["elements_processed"] == expected_elements
        assert mode["tiles_processed"] == proof["tile_visit_count"]
        assert mode["passes_completed"] == configuration["passes"]
        assert mode["slot_count"] >= 1
        assert mode["stable_addresses_verified"] is True
        assert mode["full_verification_completed"] is True
        assert mode["matches_cpu"] is True
        assert mode["mismatch_count"] == 0
        assert mode["first_mismatch_byte_offset"] is None

    for mode, handle_count, slot_count in (
        (reference, 1, 1),
        (pipeline, configuration["effective_window_slots"], 2),
    ):
        assert mode["mapping_count"] == proof["tile_visit_count"]
        assert mode["unmap_count"] == proof["tile_visit_count"]
        assert mode["event_boundary_count"] == proof["tile_visit_count"]
        assert mode["remap_count"] == proof["tile_visit_count"] - handle_count
        assert mode["remap_timing"]["sample_count"] == mode["remap_count"]
        assert mode["physical_handle_count"] == handle_count
        assert mode["max_concurrent_mappings"] == slot_count
        assert mode["slot_count"] == slot_count
        assert mode["set_access_count"] == mode["mapping_count"]
        assert mode["handle_reuse_count"] == mode["remap_count"]
        assert mode["unsafe_remap_count"] == 0

    digest = proof["cpu_reference_digest128"]
    assert len(digest) == 32
    for mode in (reference, pipeline):
        assert mode["expected_digest128"] == digest
        assert mode["output_digest128"] == digest
    for flag in (
        "handles_reused",
        "physical_window_smaller",
        "stable_virtual_addresses_verified",
        "event_boundaries_verified",
        "reference_matches_cpu",
        "pipeline_matches_cpu",
        "modes_match",
    ):
        assert proof[flag] is True
    assert all(value is True for value in report["cleanup"].values())


def main() -> int:
    schema_path = Path(sys.argv[1])
    fixture_executable = sys.argv[2]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    completed = run_fixture(fixture_executable, "vmm-poc-completed")
    assert not validation_errors(validator, completed)
    assert completed["schema_version"] == 1
    assert completed["report_type"] == "xvram.vmm_poc"
    assert completed["modes"]["backing"] == "pageable_host"
    assert completed["modes"]["traversal"] == "alternating_sequential"
    assert completed["modes"]["verification"] == "full_uint32_cpu_reference"
    assert completed["modes"]["timeout_enforcement"] == "isolated_worker"
    assert completed["modes"]["pipeline_speedup"] > 0
    assert completed["device"]["wddm_available_bytes_minimum"] <= completed["device"][
        "wddm_available_bytes_start"
    ]
    assert completed["device"]["uuid"] is None
    assert completed["device"]["luid"] is None
    assert completed["device"]["pci_bus_id"] is None
    assert_no_raw_virtual_addresses(completed)
    assert_completed_semantics(completed)

    skipped = run_fixture(fixture_executable, "vmm-poc-skipped")
    assert not validation_errors(validator, skipped)
    assert skipped["outcome"]["status"] == "skipped"
    assert skipped["outcome"]["reason"] == "device_unavailable"
    assert skipped["outcome"]["exit_code"] == 23
    assert skipped["device"] is None
    assert skipped["modes"]["reference"]["status"] == "not_run"
    assert skipped["modes"]["pipeline"]["status"] == "not_run"
    assert all(value is True for value in skipped["cleanup"].values())
    assert_no_raw_virtual_addresses(skipped)

    corruption = run_fixture(fixture_executable, "vmm-poc-corruption")
    assert not validation_errors(validator, corruption)
    assert corruption["outcome"]["status"] == "failed"
    assert corruption["outcome"]["reason"] == "data_mismatch"
    assert corruption["outcome"]["exit_code"] == 24
    assert corruption["modes"]["pipeline"]["status"] == "failed"
    assert corruption["modes"]["pipeline"]["matches_cpu"] is False
    assert corruption["modes"]["pipeline"]["mismatch_count"] == 1
    assert corruption["modes"]["pipeline"]["first_mismatch_byte_offset"] == 4096
    assert corruption["proof"]["pipeline_matches_cpu"] is False
    assert corruption["proof"]["modes_match"] is False
    assert all(value is True for value in corruption["cleanup"].values())
    assert_no_raw_virtual_addresses(corruption)

    oom = run_fixture(fixture_executable, "vmm-poc-oom")
    assert not validation_errors(validator, oom)
    assert oom["outcome"]["status"] == "failed"
    assert oom["outcome"]["reason"] == "host_oom"
    assert oom["outcome"]["exit_code"] == 25
    assert oom["outcome"]["stage"] == "host_backing"
    assert oom["modes"]["reference"]["status"] == "not_run"
    assert oom["modes"]["pipeline"]["status"] == "not_run"
    assert all(value is True for value in oom["cleanup"].values())
    assert_no_raw_virtual_addresses(oom)

    reference_only = copy.deepcopy(completed)
    reference_only["configuration"]["mode"] = "reference"
    reference_only["configuration"]["effective_window_slots"] = 1
    reference_only["modes"]["pipeline"] = not_run_mode(completed["modes"]["pipeline"])
    reference_only["modes"]["pipeline_speedup"] = None
    reference_only["proof"]["resident_physical_bytes"] = completed["proof"][
        "effective_chunk_bytes"
    ]
    reference_only["proof"]["pinned_staging_bytes"] = completed["proof"][
        "effective_chunk_bytes"
    ]
    reference_only["proof"]["pipeline_matches_cpu"] = None
    reference_only["proof"]["modes_match"] = None
    assert not validation_errors(validator, reference_only)

    pipeline_only = copy.deepcopy(completed)
    pipeline_only["configuration"]["mode"] = "pipeline"
    pipeline_only["modes"]["reference"] = not_run_mode(completed["modes"]["reference"])
    pipeline_only["modes"]["pipeline_speedup"] = None
    pipeline_only["proof"]["reference_matches_cpu"] = None
    pipeline_only["proof"]["modes_match"] = None
    assert not validation_errors(validator, pipeline_only)

    identifiers = copy.deepcopy(completed)
    identifiers["configuration"]["identifiers_included"] = True
    identifiers["device"]["uuid"] = "GPU-00112233-4455-6677-8899-aabbccddeeff"
    identifiers["device"]["luid"] = "0102030405060708"
    identifiers["device"]["pci_bus_id"] = "00000000:65:00.0"
    assert not validation_errors(validator, identifiers)

    timeout = run_fixture(fixture_executable, "vmm-poc-timeout")
    assert not validation_errors(validator, timeout)
    assert timeout["outcome"]["status"] == "failed"
    assert timeout["outcome"]["reason"] == "timeout"
    assert timeout["outcome"]["exit_code"] == 26
    assert timeout["outcome"]["mode"] == "pipeline"
    assert timeout["modes"]["pipeline"]["status"] == "timed_out"
    assert timeout["cleanup"]["complete"] is None
    assert timeout["cleanup"]["events_destroyed"] is None
    assert timeout["cleanup"]["worker_terminated"] is True
    assert_no_raw_virtual_addresses(timeout)

    cleanup_failure = run_fixture(fixture_executable, "vmm-poc-cleanup-failure")
    assert not validation_errors(validator, cleanup_failure)
    assert cleanup_failure["outcome"]["status"] == "failed"
    assert cleanup_failure["outcome"]["reason"] == "cleanup_error"
    assert cleanup_failure["outcome"]["exit_code"] == 27
    assert cleanup_failure["outcome"]["stage"] == "cleanup"
    assert cleanup_failure["modes"]["reference"]["status"] == "completed"
    assert cleanup_failure["modes"]["pipeline"]["status"] == "completed"
    assert cleanup_failure["cleanup"]["complete"] is False
    assert cleanup_failure["cleanup"]["mappings_removed"] is False
    assert cleanup_failure["cleanup"]["virtual_reservation_released"] is False
    assert_no_raw_virtual_addresses(cleanup_failure)

    redacted = run_fixture(fixture_executable, "vmm-poc-identifiers-redacted")
    assert not validation_errors(validator, redacted)
    assert redacted["configuration"]["identifiers_included"] is False
    assert redacted["device"]["uuid"] is None
    assert redacted["device"]["luid"] is None
    assert redacted["device"]["pci_bus_id"] is None

    included = run_fixture(fixture_executable, "vmm-poc-identifiers-included")
    assert not validation_errors(validator, included)
    assert included["configuration"]["identifiers_included"] is True
    assert included["device"]["uuid"] == "GPU-00112233-4455-6677-8899-aabbccddeeff"
    assert included["device"]["luid"] == "0102030405060708"
    assert included["device"]["pci_bus_id"] == "00000000:65:00.0"

    invalid = copy.deepcopy(completed)
    invalid["proof"]["handles_reused"] = False
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["configuration"]["effective_window_slots"] = 1
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(reference_only)
    invalid["configuration"]["effective_window_slots"] = 2
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(pipeline_only)
    invalid["configuration"]["effective_window_slots"] = 1
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["configuration"]["requested_window_slots"] = 9
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["modes"]["pipeline"]["physical_handle_count"] = 0
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["modes"]["pipeline"]["full_verification_completed"] = False
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["modes"]["pipeline"]["expected_digest128"] = "b" * 31
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["modes"]["pipeline"] = not_run_mode(completed["modes"]["pipeline"])
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(reference_only)
    invalid["modes"]["pipeline"] = copy.deepcopy(completed["modes"]["pipeline"])
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(reference_only)
    invalid["modes"]["pipeline_speedup"] = 1.0
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(reference_only)
    invalid["proof"]["pipeline_matches_cpu"] = True
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["cleanup"]["events_destroyed"] = False
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["device"]["uuid"] = "GPU-private-identifier"
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["device"]["luid"] = "0102030405060708"
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(timeout)
    invalid["outcome"]["reason"] = None
    assert validation_errors(validator, invalid)

    invalid = copy.deepcopy(completed)
    invalid["raw_va"] = 0x1234
    assert validation_errors(validator, invalid)

    semantic_mismatch = copy.deepcopy(completed)
    semantic_mismatch["modes"]["pipeline"]["output_digest128"] = "c" * 32
    assert not validation_errors(validator, semantic_mismatch)
    try:
        assert_completed_semantics(semantic_mismatch)
    except AssertionError:
        pass
    else:
        raise AssertionError("cross-mode digest mismatch was not detected")

    text_report = subprocess.run(
        [fixture_executable, "vmm-poc-text"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "VMM proof of concept" in text_report
    assert "reference: completed" in text_report
    assert "pipeline: completed" in text_report
    assert "slots/access/reuses:" in text_report
    assert "stable/full/matches:  yes/yes/yes" in text_report
    assert "digest128:" in text_report
    assert "Cleanup: yes" in text_report
    assert "allocations/mappings/handle:   yes/yes/yes" in text_report
    assert "reservation/staging/backing:   yes/yes/yes" in text_report
    assert "context/worker:                yes/yes" in text_report
    assert "GPU UUID:" not in text_report
    assert "GPU LUID:" not in text_report
    assert "GPU PCI:" not in text_report

    identifier_text = subprocess.run(
        [fixture_executable, "vmm-poc-text-identifiers"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "GPU UUID:  GPU-00112233-4455-6677-8899-aabbccddeeff" in identifier_text
    assert "GPU LUID:  0102030405060708" in identifier_text
    assert "GPU PCI:   00000000:65:00.0" in identifier_text
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
