#!/usr/bin/env python3
"""Require the opt-in CUDA facade to have exactly one versioned control export."""

from __future__ import annotations

import sys

import sdk_export_contract as contract


contract.EXPECTED = {"xvram_cuda_compat_get_api"}
contract.ELF_VERSION_NODE = "XVRAM_CUDA_COMPAT_1.0"


def self_test() -> int:
    valid = "XVRAM_CUDA_COMPAT_1.0 A 0\nxvram_cuda_compat_get_api@@XVRAM_CUDA_COMPAT_1.0 T 100\n"
    if contract._validate_elf(*contract._parse_elf_exports(valid)):
        return 1
    for invalid in (
        "xvram_cuda_compat_get_api T 100\n",
        valid + "cudaMalloc T 200\n",
        valid.replace("@@XVRAM_CUDA_COMPAT_1.0", "@@OTHER_1.0"),
    ):
        if not contract._validate_elf(*contract._parse_elf_exports(invalid)):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test() if sys.argv[1:] == ["--self-test"] else contract.main())
