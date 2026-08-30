#!/usr/bin/env python3
"""Exercise the real Phase 2 controller's GPU-independent failure and usage paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def assert_no_raw_addresses(value: object) -> None:
    forbidden = {"base_address", "device_pointer", "raw_va", "virtual_address"}
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            assert_no_raw_addresses(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_raw_addresses(child)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: residency_cli_contract.py <xvram-cache-bench> <report-schema>")
        return 64

    executable = Path(sys.argv[1])
    schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    command = [
        str(executable),
        "--device",
        "2147483647",
        "--logical-size",
        "8MiB",
        "--chunk-size",
        "1MiB",
        "--cache-target",
        "4MiB",
        "--device-headroom",
        "1MiB",
        "--scenario",
        "sequential",
        "--policy",
        "clock",
        "--passes",
        "2",
        "--timeout-seconds",
        "5",
        "--no-text",
        "--compact-json",
        "--json",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    assert completed.returncode == 23, (completed.returncode, completed.stderr)
    report = json.loads(completed.stdout)
    validation_errors = list(validator.iter_errors(report))
    assert not validation_errors, [error.message for error in validation_errors]
    assert report["outcome"]["status"] in ("skipped", "failed")
    assert report["outcome"]["reason"] is not None
    assert report["outcome"]["exit_code"] == 23
    assert report["cleanup"]["worker_terminated"] is True
    assert_no_raw_addresses(report)

    usage = subprocess.run(
        [str(executable), "--policy", "invalid"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert usage.returncode == 64
    assert usage.stdout == ""
    assert "error:" in usage.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
