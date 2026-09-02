from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    return subprocess.run(
        [sys.executable, "-m", "xvram.torch_bench", *arguments],
        text=True,
        capture_output=True,
        env=environment,
        timeout=15,
        check=False,
    )


def main() -> int:
    help_result = invoke("--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--cache-target" in help_result.stdout
    assert "--sdpa-backend" in help_result.stdout

    version = invoke("--version")
    assert version.returncode == 0
    assert "0.1.0-dev" in version.stdout

    invalid_geometry = invoke("--hidden", "31", "--heads", "4")
    assert invalid_geometry.returncode == 64, invalid_geometry

    invalid_seed = invoke("--seed", "0x10000000000000000")
    assert invalid_seed.returncode == 64, invalid_seed

    invalid_size = invoke("--chunk-size", "nonsense")
    assert invalid_size.returncode == 64, invalid_size

    conflicting_stdout = invoke("--json", "-", "--trace", "-")
    assert conflicting_stdout.returncode == 64, conflicting_stdout

    conflicting_path = invoke("--json", "same.json", "--trace", "same.json")
    assert conflicting_path.returncode == 64, conflicting_path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
