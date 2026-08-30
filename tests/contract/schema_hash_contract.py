#!/usr/bin/env python3
"""Freeze the public Phase 0/1 JSON contracts while Phase 2 evolves independently."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED = (
    "f56b1131bd9a2d4cf3f18bac578e5cc6cc5d9fbeca3bec8e71afae018a711e77",
    "6299205781bbfa02a626c76eadf22b7320db88c70d0043303c9d84efd032bda4",
    "7b998a8a2b50d5e1eec056d4844e572200297a443c85e0d7ee87c75966b9feeb",
)


def main() -> int:
    if len(sys.argv) != len(EXPECTED) + 1:
        print("usage: schema_hash_contract.py <capability-v1> <capability-v2> <vmm-poc-v1>")
        return 64

    failures: list[str] = []
    for raw_path, expected in zip(sys.argv[1:], EXPECTED, strict=True):
        path = Path(raw_path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{path.name}: expected {expected}, got {actual}")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
