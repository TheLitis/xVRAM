#!/usr/bin/env python3
"""Enforce the deliberately tiny Phase 4b native export boundary."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "xvram_torch_runtime_get_api",
    "xvram_torch_runtime_scratch_alloc",
    "xvram_torch_runtime_scratch_free",
}
ELF_VERSION_NODE = "XVRAM_TORCH_RUNTIME_1.0"


def _allocator_contract():
    source = Path(__file__).with_name("torch_allocator_export_contract.py")
    spec = importlib.util.spec_from_file_location("_xvram_allocator_export_contract", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared export parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_elf_exports(output: str) -> tuple[set[str], set[str], dict[str, str]]:
    exports: set[str] = set()
    version_nodes: set[str] = set()
    symbol_versions: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        raw_symbol, symbol_type = fields[0], fields[1].upper()
        if symbol_type == "A" and raw_symbol == ELF_VERSION_NODE:
            version_nodes.add(raw_symbol)
            continue
        symbol, separator, version = raw_symbol.partition("@")
        exports.add(symbol)
        if separator:
            symbol_versions[symbol] = version.lstrip("@")
    return exports, version_nodes, symbol_versions


def _validate_elf(output: str) -> list[str]:
    exports, version_nodes, symbol_versions = _parse_elf_exports(output)
    errors: list[str] = []
    if exports != EXPECTED:
        errors.append(f"expected exports {sorted(EXPECTED)}, got {sorted(exports)}")
    if ELF_VERSION_NODE not in version_nodes:
        errors.append(f"missing ELF version node {ELF_VERSION_NODE}")
    wrong_versions = {
        symbol: symbol_versions.get(symbol)
        for symbol in EXPECTED
        if symbol_versions.get(symbol) != ELF_VERSION_NODE
    }
    if wrong_versions:
        errors.append(
            f"expected every export at version {ELF_VERSION_NODE}, got {wrong_versions}"
        )
    return errors


def _self_test() -> int:
    lines = [f"{ELF_VERSION_NODE} A 0"]
    lines.extend(f"{symbol}@@{ELF_VERSION_NODE} T 100" for symbol in sorted(EXPECTED))
    errors = _validate_elf("\n".join(lines))
    if errors:
        print("valid fixture rejected: " + "; ".join(errors))
        return 1
    bad = _validate_elf(
        f"{ELF_VERSION_NODE} A 0\n"
        f"xvram_torch_runtime_get_api@@{ELF_VERSION_NODE} T 100\n",
    )
    if not bad:
        print("incomplete fixture was accepted")
        return 1
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    if len(sys.argv) != 2:
        print("usage: torch_runtime_export_contract.py <shared-library>")
        return 64
    path = Path(sys.argv[1])
    shared = _allocator_contract()
    try:
        if sys.platform == "win32":
            exports = shared._pe_exports(path)
            errors = [] if exports == EXPECTED else [
                f"expected exports {sorted(EXPECTED)}, got {sorted(exports)}"
            ]
        else:
            nm = shared.shutil.which("nm") or shared.shutil.which("llvm-nm")
            if nm is None:
                raise RuntimeError("nm or llvm-nm is required for the ELF export contract")
            result = subprocess.run(
                [nm, "-D", "--defined-only", "--format=posix", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            errors = _validate_elf(result.stdout)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"could not inspect {path}: {error}")
        return 1
    if errors:
        print("; ".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
