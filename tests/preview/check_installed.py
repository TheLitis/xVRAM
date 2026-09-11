"""CI-only installed-example and package checks. Requires a no-GPU runner."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.install.resolve()
    env = dict(os.environ)
    env["PATH"] = str(root / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(root / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    example = root / "bin" / ("xvram-preview-gemm.exe" if os.name == "nt" else "xvram-preview-gemm")
    for argv, expected in ((["--help"], 0), (["--invalid-option"], 64), (["--run"], 23)):
        result = subprocess.run([str(example), *argv], env=env, text=True, capture_output=True, timeout=30)
        if result.returncode != expected:
            raise RuntimeError(f"example {argv}: expected {expected}, got {result.returncode}\n{result.stdout}\n{result.stderr}")
    if args.package is not None:
        if args.output is None:
            parser.error("--package needs --output")
        package = args.package.resolve()
        subprocess.run([sys.executable, str(package / "preview.py"), "verify"], check=True, timeout=90)
        # Do not put the install tree on PATH: relocation must work from package alone.
        result = subprocess.run([sys.executable, str(package / "preview.py"), "smoke", "--output", str(args.output.resolve())],
                                text=True, capture_output=True, timeout=120)
        if result.returncode != 23:
            raise RuntimeError(f"packaged no-GPU runner: expected 23, got {result.returncode}\n{result.stdout}\n{result.stderr}")
        summary = json.loads((args.output / "run.json").read_text())
        if summary["status"] != "failed" or summary["process_exit_code"] != 23:
            raise RuntimeError("no-GPU run mislabeled as a successful GPU result")
    print("Installed no-GPU contracts passed. NO GPU result claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
