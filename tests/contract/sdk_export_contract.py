#!/usr/bin/env python3
"""Require the SDK shared library to expose only xvram_get_api."""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path


EXPECTED = {"xvram_get_api"}
ELF_VERSION_NODE = "XVRAM_0.1"


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _pe_exports(path: Path) -> set[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError("library is not a PE image")
    pe = _u32(data, 0x3C)
    if data[pe : pe + 4] != b"PE\0\0":
        raise ValueError("invalid PE signature")
    coff = pe + 4
    section_count = _u16(data, coff + 2)
    optional_size = _u16(data, coff + 16)
    optional = coff + 20
    directory_offset = {0x10B: 96, 0x20B: 112}.get(_u16(data, optional))
    if directory_offset is None:
        raise ValueError("unsupported PE optional-header format")
    export_rva = _u32(data, optional + directory_offset)
    if export_rva == 0:
        return set()
    sections = optional + optional_size

    def rva_to_offset(rva: int) -> int:
        for index in range(section_count):
            section = sections + index * 40
            virtual_size = _u32(data, section + 8)
            virtual_address = _u32(data, section + 12)
            raw_size = _u32(data, section + 16)
            raw_offset = _u32(data, section + 20)
            if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                return raw_offset + rva - virtual_address
        raise ValueError(f"RVA 0x{rva:x} is outside every PE section")

    directory = rva_to_offset(export_rva)
    name_count = _u32(data, directory + 24)
    names = rva_to_offset(_u32(data, directory + 32))
    exports: set[str] = set()
    for index in range(name_count):
        name_offset = rva_to_offset(_u32(data, names + index * 4))
        terminator = data.index(b"\0", name_offset)
        exports.add(data[name_offset:terminator].decode("ascii"))
    return exports


def _parse_elf_exports(output: str) -> tuple[set[str], set[str], dict[str, str]]:
    exports: set[str] = set()
    version_nodes: set[str] = set()
    symbol_versions: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        raw_symbol = fields[0]
        symbol_type = fields[1].upper()
        # GNU/LLVM nm reports ELF symbol-version definitions as absolute dynamic symbols. Keep
        # that node separate so an unversioned xvram_get_api cannot satisfy the export contract.
        if symbol_type == "A" and raw_symbol == ELF_VERSION_NODE:
            version_nodes.add(raw_symbol)
            continue

        symbol, separator, version = raw_symbol.partition("@")
        exports.add(symbol)
        if separator:
            symbol_versions[symbol] = version.lstrip("@")
    return exports, version_nodes, symbol_versions


def _elf_exports(path: Path) -> tuple[set[str], set[str], dict[str, str]]:
    nm = shutil.which("nm") or shutil.which("llvm-nm")
    if nm is None:
        raise RuntimeError("nm or llvm-nm is required for the ELF export contract")
    result = subprocess.run(
        [nm, "-D", "--defined-only", "--format=posix", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_elf_exports(result.stdout)


def _validate_elf(
    exports: set[str], version_nodes: set[str], symbol_versions: dict[str, str]
) -> list[str]:
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
    valid = _parse_elf_exports(
        "XVRAM_0.1 A 0\n"
        "xvram_get_api@@XVRAM_0.1 T 100\n"
    )
    errors = _validate_elf(*valid)
    if errors:
        print("valid fixture rejected: " + "; ".join(errors))
        return 1

    exports, nodes, versions = _parse_elf_exports(
        "XVRAM_0.1 A 0\n"
        "xvram_get_api@@XVRAM_0.1 T 100\n"
        "unexpected_absolute A 200\n"
    )
    expected = {"xvram_get_api", "unexpected_absolute"}
    if exports != expected:
        print(f"expected parsed exports {sorted(expected)}, got {sorted(exports)}")
        return 1
    if nodes != {ELF_VERSION_NODE} or versions != {"xvram_get_api": ELF_VERSION_NODE}:
        print(f"unexpected version parser result: nodes={nodes}, versions={versions}")
        return 1

    for invalid in (
        "xvram_get_api T 100\n",
        "OTHER_1.0 A 0\nxvram_get_api@@OTHER_1.0 T 100\n",
    ):
        if not _validate_elf(*_parse_elf_exports(invalid)):
            print("invalid ELF version fixture was accepted")
            return 1
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return _self_test()
    if len(sys.argv) != 2:
        print("usage: sdk_export_contract.py <shared-library>")
        return 64
    path = Path(sys.argv[1])
    try:
        if sys.platform == "win32":
            exports = _pe_exports(path)
            errors = [] if exports == EXPECTED else [
                f"expected exports {sorted(EXPECTED)}, got {sorted(exports)}"
            ]
        else:
            errors = _validate_elf(*_elf_exports(path))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"could not inspect {path}: {error}")
        return 1
    if errors:
        print("; ".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
