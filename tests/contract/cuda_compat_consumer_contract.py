#!/usr/bin/env python3
"""Build the host-only facade; prove unsupported calls cannot silently link to CUDA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=120, env=env, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    for option in ("cmake", "source", "library", "link-library", "include", "cuda-include",
                   "cublas-include", "crt-include", "cccl-include", "generator", "configuration", "binary-root"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--platform", default="")
    parser.add_argument("--make-program", default="")
    parser.add_argument("--compiler", default="")
    parser.add_argument("--compiler-flags", default="")
    parser.add_argument("--linker-flags", default="")
    args = parser.parse_args()
    Path(args.binary_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compat-consumer-", dir=args.binary_root) as temporary:
        build = Path(temporary)
        configure = [args.cmake, "-S", args.source, "-B", str(build), "-G", args.generator,
                     f"-DCMAKE_BUILD_TYPE={args.configuration}",
                     f"-DXVRAM_COMPAT_LIBRARY={args.library}",
                     f"-DXVRAM_COMPAT_LINK_LIBRARY={args.link_library}",
                     f"-DXVRAM_INCLUDE_DIR={args.include}",
                     f"-DXVRAM_CUDA_INCLUDE_DIR={args.cuda_include}",
                     f"-DXVRAM_CUBLAS_INCLUDE_DIR={args.cublas_include}",
                     f"-DXVRAM_CUDA_CRT_HEADERS_DIR={args.crt_include}",
                     f"-DXVRAM_CUDA_CCCL_HEADERS_DIR={args.cccl_include}"]
        if args.platform:
            configure.extend(["-A", args.platform])
        if args.make_program:
            configure.append(f"-DCMAKE_MAKE_PROGRAM={args.make_program}")
        if args.compiler:
            configure.append(f"-DCMAKE_CXX_COMPILER={args.compiler}")
        configure.extend([f"-DCMAKE_CXX_FLAGS={args.compiler_flags}",
                          f"-DCMAKE_EXE_LINKER_FLAGS={args.linker_flags}"])
        cases = {"supported": None, "unsupported_runtime": "cudaMemset",
                 "unsupported_driver": "cuInit", "unsupported_cublas": "cublasDgemm",
                 "unsupported_launch": "cudaLaunchKernel",
                 "device_compilation": "XVRAM_CUDA_COMPAT_HOST_ONLY"}
        for case, expected in cases.items():
            result = run(configure + [f"-DCASE={case}"])
            if result.returncode:
                raise AssertionError(f"{case} configure failed unexpectedly:\n{result.stdout}")
            result = run([args.cmake, "--build", str(build), "--config", args.configuration])
            if expected is not None:
                if result.returncode == 0 or expected not in result.stdout:
                    raise AssertionError(f"{case} did not reject {expected}:\n{result.stdout}")
                continue
            if result.returncode:
                raise AssertionError(f"supported facade failed to build:\n{result.stdout}")
            name = "consumer.exe" if os.name == "nt" else "consumer"
            executable = next((path for path in (build / args.configuration / name, build / name)
                               if path.is_file()), None)
            assert executable is not None, "consumer executable missing"
            environment = os.environ.copy()
            environment["PATH"] = str(Path(args.library).parent) + os.pathsep + environment.get("PATH", "")
            result = run([str(executable)], env=environment)
            if result.returncode:
                raise AssertionError(f"supported no-driver consumer failed:\n{result.stdout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
