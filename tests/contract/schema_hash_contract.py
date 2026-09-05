#!/usr/bin/env python3
"""Freeze every public JSON contract completed before the active phase."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED = (
    "f56b1131bd9a2d4cf3f18bac578e5cc6cc5d9fbeca3bec8e71afae018a711e77",
    "6299205781bbfa02a626c76eadf22b7320db88c70d0043303c9d84efd032bda4",
    "7b998a8a2b50d5e1eec056d4844e572200297a443c85e0d7ee87c75966b9feeb",
    "70ef270bcff4e0b32c412d40cfd44e8b38108677de45951437ed77120d603980",
    "d2fbc19f2175f4922dc4b36b9653384b3bab40bed3e6daa345b5bdc8b084705c",
    "99c632f1511a7aa8fe57f59458c755f2cc03f3b413136ed3d95fb968d69bfe39",
    "35d6fbe489706671d3ac90c6db75d92a4af0bcd9671a01d5a0b8a7e62a8fd710",
    "cf06453a25a35193f87d639338ae27c2c65eebd8774bf6704f029d2f0df3051d",
    "1cb087f1d7e8e53552b40fa89d9bd913c5163da849cf4f794624d1214c12b7d2",
    "69367b09de552e2c6ee0a7008ba2bc44729a2977050a50a439e6efa594b2ffbd",
    "9a27115d0e67a1a1c546d98f41d9554e7245b0422ab2254e128cf607db7dec34",
    "343e4742d62b01869c259adbffa0c1d6fcf8b9a141140fe447b5ab9b6a0b03de",
)


def main() -> int:
    # Older build trees pass six or eight paths. Discover the remaining completed
    # contracts beside them; reconfiguration must not be needed to protect Phase 5.
    if len(sys.argv) not in (7, 9, len(EXPECTED) + 1):
        print(
            "usage: schema_hash_contract.py <capability-v1> <capability-v2> "
            "<vmm-poc-v1> <residency-report-v1> <residency-trace-v1> <gemm-report-v1> "
            "[<pytorch-inference-report-v1> <pytorch-inference-trace-v1> "
            "[<compression-report-v1> <compression-trace-v1> "
            "<pytorch-inference-report-v2> <pytorch-inference-trace-v2>]]"
        )
        return 64

    paths = [Path(raw_path) for raw_path in sys.argv[1:]]
    if len(paths) == 6:
        schema_dir = paths[0].parent
        paths.extend(
            (
                schema_dir / "pytorch-inference-report-v1.schema.json",
                schema_dir / "pytorch-inference-trace-record-v1.schema.json",
            )
        )

    if len(paths) == 8:
        schema_dir = paths[0].parent
        paths.extend(
            (
                schema_dir / "adaptive-compression-report-v1.schema.json",
                schema_dir / "compression-trace-record-v1.schema.json",
                schema_dir / "pytorch-inference-report-v2.schema.json",
                schema_dir / "pytorch-inference-trace-record-v2.schema.json",
            )
        )

    failures: list[str] = []
    for path, expected in zip(paths, EXPECTED, strict=True):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{path.name}: expected {expected}, got {actual}")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
