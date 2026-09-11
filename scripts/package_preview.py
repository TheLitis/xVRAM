"""Assemble a Windows preview from an already installed and tested SDK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
from preview import require, sha256, verify_package, write_json

REQUIRED_INSTALL_FILES = (
    "bin/xvram.dll", "bin/xvram_cuda_compat.dll", "bin/xvram-compat-bench.exe",
    "bin/xvram-probe.exe", "bin/xvram-preview-gemm.exe", "bin/cublas64_13.dll",
    "bin/cublasLt64_13.dll", "bin/nvcomp64_5.dll", "bin/nvcomp_cpu64_5.dll",
    "include/xvram/xvram.h", "include/xvram/cuda_compat.h",
    "lib/cmake/xVRAM/xVRAMConfig.cmake", "lib/cmake/xVRAM/xVRAMTargets.cmake",
    "share/licenses/xvram/LICENSE.cublas.txt", "share/licenses/xvram/LICENSE.nvcomp.txt",
    "share/licenses/xvram/NOTICE.nvcomp.txt", "share/licenses/xvram/LICENSE.lz4.txt",
    "share/xvram/schemas/cuda-compat-v1.schema.json",
    "share/xvram/schemas/cuda-compat-trace-v1.schema.json",
)


def assemble(source: Path, install: Path, destination: Path, revision: str, ci_url: str) -> Path:
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "full source revision required")
    require(not destination.exists(), "package output must be new")
    require(install.is_dir() and not install.is_symlink(), "missing install tree")
    for name in REQUIRED_INSTALL_FILES:
        require((install / name).is_file(), f"incomplete Windows install tree: {name}")
    for path in install.rglob("*"):
        require(not path.is_symlink(), "Windows package cannot contain install-tree symlinks")
        require(path.is_dir() or path.is_file(), "special file in install tree")
    config = json.loads((source / "release/preview.json").read_text(encoding="utf-8"))
    tag = config["tag"]
    require(re.fullmatch(r"preview-[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[1-9][0-9]*", tag) is not None,
            "invalid preview release tag")
    destination.mkdir(parents=True, exist_ok=False)
    package = destination / ("xvram-" + tag + "-windows-x64")
    shutil.copytree(install, package)
    for name in ("LICENSE", "NOTICE", "PRIVACY.md", "SECURITY.md", "VERSION", "DEVELOPMENT.md"):
        shutil.copy2(source / name, package / name)
    shutil.copy2(source / "docs/preview/README.md", package / "README.md")
    shutil.copy2(source / "scripts/preview.py", package / "preview.py")
    shutil.copy2(source / "tests/requirements.txt", package / "requirements-preview.txt")
    (package / "tools").mkdir()
    shutil.copy2(source / "tests/contract/compat_contract.py", package / "tools/compat_contract.py")
    shutil.copytree(source / "docs", package / "docs")
    shutil.copytree(source / "examples/preview-gemm", package / "examples/preview-gemm")
    for path in package.rglob("*"):
        require(not path.is_symlink(), "source symlink cannot enter package")
        require(path.suffix.lower() not in (".pdb", ".obj", ".pyc", ".dmp"), "private/build artifact in package")
        require(path.name not in (".env", ".git", "__pycache__"), "private/cache directory in package")
    files = {p.relative_to(package).as_posix(): sha256(p) for p in sorted(package.rglob("*")) if p.is_file()}
    manifest = {"format": "xvram.developer_preview.package.v1", "release_tag": tag,
                "source_commit": revision, "sdk_version": (source / "VERSION").read_text().strip(),
                "platform": "windows-x64", "build_type": "Release", "ci_url": ci_url,
                "gpu_validation": "not_run_by_packaging_ci",
                "scope": "explicit synchronous FP32 CUDA/cuBLAS; compression disabled in this profile",
                "files": files}
    write_json(package / "package-manifest.json", manifest)
    verify_package(package)
    archive = destination / (package.name + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                stream.write(path, path.relative_to(destination).as_posix())
    (destination / "SHA256SUMS.txt").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--ci-url", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        require(revision == args.source_commit, "checkout differs from requested revision")
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True).strip()
        require(not dirty, "refusing a dirty tracked source tree")
        require(re.fullmatch(r"https://github\.com/TheLitis/xVRAM/actions/runs/[0-9]+", args.ci_url) is not None,
                "package must link to its xVRAM CI run")
        print(assemble(source, args.install.resolve(), args.output.resolve(), revision, args.ci_url))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
