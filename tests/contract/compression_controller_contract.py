#!/usr/bin/env python3
"""Validate controller-owned failure reports after real child termination and reap."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import jsonschema

from compression_contract import contains_private_runtime_value


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: compression_controller_contract.py <schema> <controller-tests> <worker>")
        return 64
    schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    result = subprocess.run(
        [sys.argv[2], sys.argv[3], "--emit-failure-reports"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    # The native test also checks every marked child's PID is gone before emitting its report.
    assert result.returncode == 0, (result.returncode, result.stderr)
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(reports) == 8, len(reports)
    assert Counter(report["outcome"]["exit_code"] for report in reports) == {
        26: 2, 27: 4, 74: 2
    }
    for report in reports:
        validator.validate(report)
        assert not contains_private_runtime_value(report)
        outcome = report["outcome"]
        assert outcome["status"] == ("timeout" if outcome["exit_code"] == 26 else "failed")
        assert report["cleanup"]["worker_terminated"] is True
        assert report["telemetry"]["trace_complete"] is False
        assert report["cleanup"]["trace_closed"] is (outcome["exit_code"] != 74)
        assert report["telemetry"]["trace_records_dropped"] == (
            1 if outcome["exit_code"] == 74 else 0
        )
        for name, value in report["cleanup"].items():
            if name not in {"trace_closed", "worker_terminated"}:
                assert value is None, (name, value)
        assert all(value is None for value in report["proof"].values())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
