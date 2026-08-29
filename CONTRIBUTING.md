# Contributing

xVRAM is in an early architecture-validation stage. Please open an issue before a large
change so the working-set and correctness contracts can be agreed first.

## Development checks

```text
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Changes should keep the no-CUDA-driver path functional, add tests for new state or
serialization behavior, and update the report schema documentation when fields change.
Performance changes need correctness results and a reproducible baseline.

Commit messages should be imperative and focused. Do not commit generated build output,
probe reports containing machine identifiers, or profiler captures.
