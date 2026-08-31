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


def _parse_elf_exports(output: str) -> set[str]:
    exports: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        # GNU/LLVM nm reports ELF symbol-version definitions as absolute dynamic symbols (for
        # example ``XVRAM_0.1 A 0``). They are linker metadata, not callable/data exports.
        symbol = fields[0].split("@", 1)[0]
        if fields[1].upper() == "A" and symbol == ELF_VERSION_NODE:
            continue
        exports.add(symbol)
    return exports


def _elf_exports(path: Path) -> set[str]:
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


def _self_test() -> int:
    exports = _parse_elf_exports(
        "XVRAM_0.1 A 0\n"
        "xvram_get_api@@XVRAM_0.1 T 100\n"
        "unexpected_absolute A 200\n"
    )
    expected = {"xvram_get_api", "unexpected_absolute"}
    if exports != expected:
        print(f"expected parsed exports {sorted(expected)}, got {sorted(exports)}")
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
        exports = _pe_exports(path) if sys.platform == "win32" else _elf_exports(path)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"could not inspect {path}: {error}")
        return 1
    if exports != EXPECTED:
        print(f"expected exports {sorted(EXPECTED)}, got {sorted(exports)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
