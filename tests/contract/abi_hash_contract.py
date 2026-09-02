#!/usr/bin/env python3
"""Freeze the completed Phase 3 and Phase 4a public C headers."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED = {
    "xvram.h": "dbc4162937b49f623382d6567747276dbb84f0164c008a816d335b1a2b7e5357",
    "torch_allocator.h": "9269cfccc81b2fca119438f1c62ab34299422d9160d357935a816d5a1bfc440f",
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: abi_hash_contract.py <xvram.h> <torch_allocator.h>")
        return 64
    failures: list[str] = []
    for raw_path in sys.argv[1:]:
        path = Path(raw_path)
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
