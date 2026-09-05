#!/usr/bin/env python3
"""Freeze every native ABI header completed through Phase 5."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED = {
    "xvram.h": "dbc4162937b49f623382d6567747276dbb84f0164c008a816d335b1a2b7e5357",
    "torch_allocator.h": "9269cfccc81b2fca119438f1c62ab34299422d9160d357935a816d5a1bfc440f",
    "torch_runtime.h": "df1cdd104b5d3f90bc7ec6f6fddd7bc994ad18ee3b4440587d0a26e4d7b36b5e",
    "xvram_v2.h": "ded7362a90d12476f20fff22cf8f012232fd2b4b1f68467113fb01279d35f5a2",
    "torch_runtime_v2.h": "6d48811e0d91db0c90d9893ed93120668e61a40b73724243e84c50501fd5cff8",
}


def main() -> int:
    if len(sys.argv) not in (3, 4, 6):
        print(
            "usage: abi_hash_contract.py <xvram.h> <torch_allocator.h> "
            "[<internal/torch_runtime.h> [<xvram_v2.h> <internal/torch_runtime_v2.h>]]"
        )
        return 64

    paths = [Path(raw_path) for raw_path in sys.argv[1:]]
    if len(paths) == 2:
        paths.append(paths[0].parent / "internal" / "torch_runtime.h")
    if len(paths) == 3:
        paths.extend(
            (paths[0].parent / "xvram_v2.h",
             paths[0].parent / "internal" / "torch_runtime_v2.h")
        )

    failures: list[str] = []
    for path in paths:
        expected = EXPECTED.get(path.name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is None or actual != expected:
            failures.append(f"{path.name}: expected {expected}, got {actual}")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
