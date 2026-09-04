#!/usr/bin/env python3
"""Validate production C++ compression report/trace serialization against v1 schemas."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema

from compression_contract import contains_private_runtime_value, semantic_errors


def invoke(fixture: Path, mode: str) -> str:
    result = subprocess.run(
        [str(fixture), mode],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"C++ compression fixture {mode} exited {result.returncode}: {result.stderr}"
        )
    if not result.stdout.endswith("\n"):
        raise AssertionError(f"C++ compression fixture {mode} is not newline terminated")
    return result.stdout


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: compression_cpp_contract.py <report-schema> <trace-schema> <fixture>"
        )
        return 64

    report_schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    trace_schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    fixture = Path(sys.argv[3])
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.Draft202012Validator.check_schema(trace_schema)

    report = json.loads(invoke(fixture, "--emit-json-fixture"))
    jsonschema.Draft202012Validator(report_schema).validate(report)
    if errors := semantic_errors(report):
        raise AssertionError(f"C++ completed report failed semantics: {errors}")
    if contains_private_runtime_value(report):
        raise AssertionError("C++ completed report exposes a CUDA address, pointer, or stream")

    trace = json.loads(invoke(fixture, "--emit-trace-fixture"))
    jsonschema.Draft202012Validator(trace_schema).validate(trace)
    if contains_private_runtime_value(trace):
        raise AssertionError("C++ trace exposes a CUDA address, pointer, or stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
