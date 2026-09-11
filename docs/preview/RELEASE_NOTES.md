# xVRAM Developer Preview 1

A Windows x64 distribution of the existing explicit CUDA/cuBLAS integration,
with a public-API C++ example and a bounded, evidence-producing demo workflow.

## Included

- Release SDK and existing CLI benchmarks, app-local pinned cuBLAS/nvCOMP libraries,
  public headers, CMake targets, licenses and notices.
- `xvram-preview-gemm`: small standalone FP32 example, full CPU FP64 reference,
  explicit error/cleanup handling, using only installed public C API headers.
- `preview.py verify`, `smoke`, `oversubscribe`, `validate`: integrity checks,
  real isolated runs and the established Phase 6a report/trace validators.
- New result directory for every run, provenance and file hashes, stdout/stderr,
  raw JSON/trace and offline HTML. Missing/failed GPU runs never count as passes.
- Documented Astra boundary at `98e41acc`, evidence limitations, conceptual gallery
  asset and Product Hunt submission kit in `docs/preview`.

Download the Windows ZIP and `SHA256SUMS.txt`, then read its `README.md`.
Python 3.10+ with the pinned validator dependency, a compatible NVIDIA GPU/driver,
Visual C++ v14 x64 runtime and sufficient available RAM are required. No OpenAI API
key or automatic telemetry upload is involved.

## Validation boundary

Publication requires the existing Windows/Linux CI matrix and separate preview
unit, build, installed-example and packaging checks on the same source revision.
CI has no GPU: **this is not fresh RTX 3070 hardware certification**. Historical
maintainer results are linked and labeled. Record a new run of this exact package
before presenting new measurements. Unit-test fixtures are not hardware evidence.

## Limitations

The production residency architecture, ABI and frozen schemas are unchanged.
This profile is explicit, synchronous FP32 integration with compression disabled:
not unchanged-executable interception, arbitrary kernels, general PyTorch/LLM
support, training or a universal speedup. Phase 6b proof gates remain unproven.
The SDK stays `0.1.0-dev` and is not for production workloads.

xVRAM is Apache-2.0. Bundled components retain their own terms. Publishing this
release is not a Product Hunt submission or an assertion of organizer approval.
