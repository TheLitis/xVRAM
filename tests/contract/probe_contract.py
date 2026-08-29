#!/usr/bin/env python3
"""GPU-independent contract smoke test for xvram-probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def main() -> int:
    executable = sys.argv[1]
    schema_path = Path(sys.argv[2])
    fixture_executable = sys.argv[3]

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    fixture = subprocess.run(
        [fixture_executable],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    fixture_report = json.loads(fixture.stdout)
    fixture_errors = sorted(
        validator.iter_errors(fixture_report), key=lambda item: list(item.absolute_path)
    )
    assert not fixture_errors, fixture_errors
    assert fixture_report["cuda"]["devices"][0]["vmm_smoke"]["remap_copy_verified"] is True
    assert fixture_report["cuda"]["devices"][0]["dxgi"]["local"]["budget_bytes"] > 0
    assert fixture_report["transfer_benchmark"]["device_ordinal"] == 0
    assert fixture_report["overlap_benchmark"]["status"] == "completed"
    assert fixture_report["overlap_benchmark"]["compute_verified"] is True
    assert fixture_report["overlap_benchmark"]["transfer_verified"] is True
    assert fixture_report["overlap_benchmark"]["h2d"]["speedup"] > 1.0

    completed = subprocess.run(
        [executable, "--json", "-", "--no-text", "--compact-json", "--skip-vmm-smoke"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        return completed.returncode

    report = json.loads(completed.stdout)
    errors = sorted(validator.iter_errors(report), key=lambda item: list(item.absolute_path))
    if errors:
        for validation_error in errors:
            location = ".".join(str(part) for part in validation_error.absolute_path) or "<root>"
            print(f"schema error at {location}: {validation_error.message}", file=sys.stderr)
        return 1

    assert report["schema_version"] == 2
    assert report["report_type"] == "xvram.capability_probe"
    assert report["build"]["version"]
    assert report["build"]["cuda_headers_version"] >= 13000
    assert report["system"]["architecture"]
    assert isinstance(report["cuda"]["library_loaded"], bool)
    assert isinstance(report["cuda"]["nvml_library_loaded"], bool)
    assert isinstance(report["cuda"]["devices"], list)
    assert isinstance(report["diagnostics"], list)
    assert report["overlap_benchmark"] is None
    if not report["cuda"]["library_loaded"]:
        assert report["cuda"]["devices"] == []
        assert any(
            item["component"] == "cuda" and item["operation"] == "load_driver"
            for item in report["diagnostics"]
        )

    for device in report["cuda"]["devices"]:
        assert device["uuid"] is None
        assert device["luid"] is None
        assert device["pci_bus_id"] is None
        assert "pci_domain_id" not in device["attributes"]
        assert "pci_bus_id" not in device["attributes"]
        assert "pci_device_id" not in device["attributes"]
        assert device["driver_model"] in ("wddm", "tcc", "mcdm", "unknown", None)
        assert "virtual_memory_management" in device["capabilities"]
        assert "allocation_granularity" in device
        assert "vmm_smoke" in device
        if device["dxgi"] is not None:
            assert device["dxgi"]["luid"] is None

    invalid_device = subprocess.run(
        [executable, "--device", "2147483647", "--no-text", "--json", "-", "--compact-json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert invalid_device.returncode == 21

    invalid_overlap = subprocess.run(
        [executable, "--overlap-samples", "2"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert invalid_overlap.returncode == 64

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
