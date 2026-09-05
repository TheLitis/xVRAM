"""Verify a relocated installation loads the configured app-local cuBLAS pair without a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--configuration", default="Release")
    args = parser.parse_args()
    provenance = json.loads((args.build / "compat-cublas-provenance.json").read_text())
    name = "xvram-cuda-compat-load-helper" + (".exe" if os.name == "nt" else "")
    helpers = [p for p in args.build.rglob(name) if "_deps" not in p.parts]
    configured = [p for p in helpers if p.parent.name == args.configuration]
    if configured:
        helpers = configured
    assert len(helpers) == 1, helpers
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    with tempfile.TemporaryDirectory(prefix="xvram-compat-relocated-") as temporary:
        root = Path(temporary) / "package with spaces"
        shutil.copytree(args.install, root)
        library_dir = root / provenance["library_directory"]
        for role in ("core", "lt"):
            library = library_dir / provenance[role + "_name"]
            assert digest(library) == provenance[role + "_sha256"], library
        helper = library_dir / name
        shutil.copy2(helpers[0], helper)
        observed = subprocess.run([str(helper)], env=environment, text=True,
                                  capture_output=True, timeout=60, check=True)
        info = json.loads(observed.stdout)
        assert info["source"] == "app_local" and info["lt_version"] > 0, info
        executable = root / "bin" / ("xvram-compat-bench" + (".exe" if os.name == "nt" else ""))
        subprocess.run([str(executable), "--version"], env=environment, timeout=30, check=True)
    print("Relocated app-local cuBLAS core/Lt hashes and dispatch verified")


if __name__ == "__main__":
    main()
