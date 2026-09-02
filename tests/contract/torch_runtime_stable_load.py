#!/usr/bin/env python3
"""Load the Phase 4b bridge and verify its PyTorch Stable-ABI registration."""

from __future__ import annotations

import sys
from pathlib import Path


EXPECTED_SCHEMA = (
    "xvram_internal::_wrap_resolved_v1(int session, int lease, int allocation, "
    "int offset, int[] sizes, int[] strides, int scalar_type) -> Tensor"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: torch_runtime_stable_load.py <shared-library>")
        return 64
    try:
        import torch

        torch.ops.load_library(str(Path(sys.argv[1]).resolve()))
        operation = torch.ops.xvram_internal._wrap_resolved_v1.default
        schema = str(operation._schema)
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        print(f"could not load the PyTorch Stable-ABI bridge: {error}")
        return 1
    if schema != EXPECTED_SCHEMA:
        print(f"unexpected Stable-ABI schema: {schema}")
        return 1
    try:
        operation(1, 1, 1, 0, [1], [1], 6)
    except RuntimeError as error:
        if "xVRAM" not in str(error):
            print(f"invalid-handle wrapper returned an unexpected error: {error}")
            return 1
    else:
        print("invalid native handles unexpectedly produced a tensor")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
