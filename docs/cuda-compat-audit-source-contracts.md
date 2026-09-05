# Source evidence for the unchanged CUDA backend

This is a bounded continuation of Phase 6b.0, not an interceptor. It closes the
"which pinned source definitions should be reviewed?" gap while retaining
**NO-GO** for transparent execution. Neither a matching source hash nor a familiar
kernel symbol proves the binary's compiled argument ABI or the memory ranges of a
captured launch. Production residency and all existing ABI/report contracts remain
unchanged.

## Reproducible offline index

The packaged `compat_audit_source_profile.json` pins the byte count and SHA-256 of
18 selected upstream source files at commit
`6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`. They total less than 2 MiB and are
diagnostic dependencies, not a source checkout to rebuild. Obtain or verify them
outside the xVRAM repository:

```powershell
./scripts/fetch-phase6b-audit-sources.ps1 -SourceRoot D:/xVRAM-dependencies/phase6b0/llama-source
./scripts/fetch-phase6b-audit-sources.ps1 -SourceRoot D:/xVRAM-dependencies/phase6b0/llama-source -VerifyOnly
```

The downloader fetches only fixed-commit raw GitHub URLs, verifies before admitting
each file, rejects incorrect existing files, and refuses source paths inside the
repository or through reparse points. It does not download models, compile code,
load CUDA or change installed libraries.

After setting `PYTHONPATH=python` or installing the Python package:

```text
python -m xvram.compat_audit_sources --source-root D:/xVRAM-dependencies/phase6b0/llama-source --report artifacts/phase6b0-rtx3070-20260906/review/14b-ub1-cupti-report.json --report artifacts/phase6b0-rtx3070-20260906/review/14b-ub128-cupti-report.json --report artifacts/phase6b0-rtx3070-20260906/review/32b-ub1-cupti-report.json --report artifacts/phase6b0-rtx3070-20260906/review/32b-ub128-cupti-report.json --json artifacts/phase6b0-rtx3070-20260906/review/source-index-v1.json
```

`--report` is repeatable. The index reads `coverage.activity_kernel_types`, not
Runtime/Driver launch-callback counts. It checks consumed report fields and count
reconciliation; full validation against the frozen audit schema remains a separate
gate. It never rewrites reports, sources or an existing output. `--json -` writes to
stdout. Exit 0 means that the index was produced, **not GO**; input rejection is 23
and output failure is 74. The separate internal report type is
`xvram.cuda_compat_source_index` v1; the original audit report/trace schemas are not
extended.

Inputs are bounded to 16 reports, 8 MiB per report, 4,096 distinct activity names,
32 source files and 2 MiB per source file. Duplicate report hashes, duplicate JSON
keys, invalid counters, overflow, inconsistent activity totals and malformed/private
kernel names are rejected. Output has only source-relative paths, immutable URLs,
hashes and normalized names/counters, never local paths, CUDA addresses or handles.
The input reports and their unrelated fields are not copied to the output.

## What the existing captures actually cover

The four immutable reviewed CUPTI reports contain 72,262 GPU kernel activities and
39 distinct names in total. Per-case totals remain:

| Native CPU-offload case | Activity names | GPU activities |
| --- | ---: | ---: |
| 14B, microbatch 1 | 15 | 26,860 |
| 14B, microbatch 128 | 39 | 8,776 |
| 32B, microbatch 1 | 14 | 26,758 |
| 32B, microbatch 128 | 39 | 9,868 |

The initial derived source index finds a unique candidate definition for **35 of 39
names**, across 17 top-level function identifiers. Four cuBLAS/CUTLASS-style library
names remain opaque. This is candidate coverage, not 35 verified ABI contracts:
**binary-bound types remain zero**. No absent family in these eight-GPU-layer
CPU-offload captures is assumed absent from a future all-GPU model.

| Observed candidate family | Pinned source review entry | Outstanding memory evidence |
| --- | --- | --- |
| Quantized matvec and fused variants | `mmvq.cu`, `common.cuh` | Q4_K/Q6_K and q8 layouts; fusion struct pointers; indices and strides |
| Quantized prefill and stream-K fixup | `mmq.cuh`, `quantize.cu` | Padding, per-tile accesses, global fixup scratch and launch ordering |
| Float matvec/matmul and conversion | `mmvf.cu`, `mmf.cuh`, `convert.cu` | Dtypes, optional indices, exact strided extents and tail accesses |
| KV writes/reads, RoPE | `set-rows.cu`, `getrows.cu`, `rope.cu` | Device index values, destination rows, aliases and KV generations |
| RMS norm, softmax, broadcast, gated unary, copy | `norm.cu`, `softmax.cu`, `binbcast.cu`, `unary.cu`, `cpy.cu` | Fused operands, masks, strides, padding and aliases |
| Batched pointer construction | `ggml-cuda.cu` | Device-generated pointer tables and indirect cuBLAS operand ranges |
| Four library kernels | No verified source-to-binary binding | Public cuBLAS call contract, workspace and exact opaque-call ordering |

The matcher is deliberately a small top-level identifier recognizer, not a C++
parser or full demangler. Template arguments and mangled names are preserved.
Multiple definitions remain ambiguous; missing or altered source cannot supply
candidate evidence. Every kernel retains explicit proof obligations and false
binary-binding/memory-bound flags regardless of incoming claims.

## Source-only facts that constrain the next design

The index verifies each file's pinned hash and unique anchor locations before
emitting these manually reviewed facts. Source presence is proven; selection,
compiled layout and concrete launch bindings are not.

- The native [VMM pool](https://github.com/ggml-org/llama.cpp/blob/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/ggml/src/ggml-cuda/ggml-cuda.cu#L538)
  has a 32 GiB VA arena and 128-byte LIFO suballocations. It appends physical mappings
  and immediately releases their handles, later unmapping the aggregate mapped
  extent. Reservation, handle and physical mapping lifetimes must be separate in
  the observer; native map/unmap *call counts* need not match.
- [Batched pointer construction](https://github.com/ggml-org/llama.cpp/blob/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/ggml/src/ggml-cuda/ggml-cuda.cu#L1341)
  writes pointer tables on GPU for cuBLAS. An allocation containing pointers is not
  an operand-range contract. Account for the table writer, indirect operands and
  consumer ordering.
- The [fusion struct](https://github.com/ggml-org/llama.cpp/blob/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/ggml/src/ggml-cuda/common.cuh#L1545)
  carries gate, bias and scale pointers. Quantized matvec can select channels through
  device indices. A two-input matrix model misses these possible reads.
- [q8 quantization](https://github.com/ggml-org/llama.cpp/blob/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/ggml/src/ggml-cuda/quantize.cu#L54)
  distinguishes valid source columns from padded output. The GGML enum maps 12 to
  Q4_K and 14 to Q6_K, but those template values do not provide actual launch bounds.
- [Stream-K MMQ](https://github.com/ggml-org/llama.cpp/blob/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/ggml/src/ggml-cuda/mmq.cuh#L1443)
  conditionally allocates global fixup storage in the native pool and launches a
  follow-up kernel. Dynamic shared-memory bytes alone do not bound its scratch.

## Next proof obligations, not authorization to intercept

For every selected module/function generation, establish the compiled argument
layout and bind the relevant source operation, runtime scalar values, tensor
subranges/aliases, indirect targets, global scratch and complete event ordering.
Then compute overflow-safe chunk-rounded working sets against native pool,
cuBLAS workspace and live WDDM headroom. Opaque library kernels should be reasoned
about through a verified public operation boundary, not guessed private parameters.

Full dynamic API routing, terminal collection and coverage of loading, warmup,
prefill, repeated decode and teardown remain separate mandatory evidence gates.
The source index cannot repair them or approve a backend rebuild, kernel
replacement, hooks or a different GGML plugin. It is a reproducible, reviewable map
of the next obligations, and its verdict is intentionally always NO-GO.

Focused no-driver tests:

```text
python -m unittest discover -s tests/python -p test_compat_audit_sources.py
```
