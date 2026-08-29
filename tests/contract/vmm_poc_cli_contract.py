#!/usr/bin/env python3
"""No-GPU CLI/controller contract check for xvram-vmm-poc."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def run_and_validate(
    command: list[str],
    schema: dict[str, object],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    completed = subprocess.run(
        [
            *command,
            "--device",
            "2147483647",
            "--json",
            "-",
            "--compact-json",
            "--no-text",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
        env=env,
    )
    assert completed.returncode == 23, completed
    assert not completed.stderr, completed.stderr
    assert completed.stdout.count("\n") == 1, "compact JSON must occupy one line"
    report = json.loads(completed.stdout)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report),
        key=lambda error: list(error.absolute_path),
    )
    assert not errors, "\n".join(
        f"{list(error.absolute_path)}: {error.message}" for error in errors
    )
    assert report["report_type"] == "xvram.vmm_poc"
    assert report["outcome"]["status"] == "skipped"
    assert report["outcome"]["reason"] == "device_unavailable"
    assert report["cleanup"]["complete"] is True
    assert report["cleanup"]["worker_terminated"] is True
    assert report["device"] is None


def main() -> int:
    executable = Path(sys.argv[1]).resolve()
    schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    run_and_validate([str(executable)], schema)

    path_environment = os.environ.copy()
    path_environment["PATH"] = str(executable.parent) + os.pathsep + path_environment.get(
        "PATH", ""
    )
    with tempfile.TemporaryDirectory(prefix="xvram-cli-contract-") as temporary_directory:
        path_command = (
            ["cmd.exe", "/d", "/s", "/c", executable.name]
            if os.name == "nt"
            else [executable.name]
        )
        run_and_validate(
            path_command,
            schema,
            cwd=temporary_directory,
            env=path_environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
