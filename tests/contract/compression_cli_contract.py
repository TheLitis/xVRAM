#!/usr/bin/env python3
"""Exercise the installed-facing xvram-compression-bench CLI contract without a proof run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema


def invoke(executable: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compression_cli_contract.py <executable> <report-schema>")
        return 64
    executable = Path(sys.argv[1])
    schema = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    help_result = invoke(executable, "--help")
    assert help_result.returncode == 0, help_result.stderr
    for option in (
        "--compression-policy",
        "--path",
        "--host-store-cap",
        "--codec-slots",
        "--codec-workers",
        "--compression-scratch-cap",
    ):
        assert option in help_result.stdout, option

    version = invoke(executable, "--version")
    assert version.returncode == 0, version.stderr
    assert "0.1.0-dev" in version.stdout

    for arguments in (
        ("--codec-slots", "1"),
        ("--passes", "1"),
        ("--no-text",),
        ("--json", "-", "--trace", "-"),
    ):
        invalid = invoke(executable, *arguments)
        assert invalid.returncode == 64, (arguments, invalid.returncode, invalid.stderr)

    # Four bytes are deliberately too small to prove VRAM oversubscription on any supported
    # CUDA device. This gives both GPU and no-driver CI a bounded production controller/report
    # path without allocating the logical backing store.
    skipped = invoke(
        executable,
        "--logical-size",
        "4B",
        "--timeout-seconds",
        "5",
        "--compact-json",
        "--no-text",
        "--json",
        "-",
    )
    assert skipped.returncode == 23, (skipped.returncode, skipped.stderr)
    report = json.loads(skipped.stdout)
    jsonschema.Draft202012Validator(schema).validate(report)
    assert report["schema_version"] == 1
    assert report["report_type"] == "xvram.adaptive_compression"
    assert report["outcome"]["status"] == "skipped"
    assert report["outcome"]["exit_code"] == 23
    assert report["cleanup"]["worker_terminated"] is True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
