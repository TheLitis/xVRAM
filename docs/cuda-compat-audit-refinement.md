# Phase 6b.0 evidence refinement

This continuation closes bounded observation gaps, not the full transparent-execution
gate. The result remains **NO-GO**: the original llama.cpp backend has not been replaced,
intercepted or run through xVRAM. Production residency and all Phase 1–6a contracts
remain unchanged. The original audit report v1 and trace v1 schemas are also frozen.

## What the new evidence establishes

The collector's opt-in trace v2 records the documented fixed metadata of
`cuGetProcAddress[_v2]` and `cudaGetDriverEntryPoint[ByVersion]`: requested symbol,
version, flags, returned query status when present, and a bounded process-local ID
for a non-null result. It never invokes or replaces the result, changes arguments,
reads kernel argument arrays, or issues CUDA work. CUPTI callback restrictions remain
the [official observation boundary](https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api).
The default remains trace v1; its unknown-detail behavior is preserved.

An independent offline correlator separates Runtime callbacks, nested Driver callbacks
and GPU activities. It checks per-thread entry/exit nesting, correlation identity,
context, stream, kernel name, launch geometry, shared bytes and available API activity
records. A nested Runtime/Driver pair is not counted as two GPU executions. Buffered
activities need not appear next to their callbacks. Two bounded passes hash their
consumed bytes; changed input, ambiguous identities or contradictory metadata cannot
produce reconciled evidence. The enclosing sidecar rechecks file hashes across both
the correlation and resolver analyses.

`launch_records_reconciled:true` means **the observed records match**. It does not prove
that no record was lost, establish device ordering or tensor bounds, or repair an
incomplete footer. `trace_complete`, `device_ordering_proven` and
`semantic_ranges_proven` remain separate. The strict observations contract cannot
emit a positive interoperability decision.

The [source-evidence index](cuda-compat-audit-source-contracts.md) additionally pins
18 upstream files and verifies six reviewed source-only facts. It matches 35 of 39
observed kernel names to unique candidate definitions, leaving four library types
opaque and **zero** proven binary/argument bindings. Important constraints include
GPU-generated cuBLAS pointer tables, fused operands, padded quantization, global
stream-K scratch, and the native VMM arena's distinct reservation/mapping/handle
lifetimes. Native map/unmap *call-count* equality is not a valid arena invariant.

## Reproduction

Use the pinned binaries/models from the [main audit guide](cuda-compat-audit.md),
rebuild the diagnostic collector, and choose a new output directory:

```powershell
python -m xvram.compat_audit --stage all --trace-version 2 `
  --model 14b --microbatch 128 --gpu-layers 8 `
  --binary-dir D:/xVRAM-dependencies/phase6b0/llama-b10819 `
  --model-dir D:/xVRAM-dependencies/phase6b0/qwen14b `
  --capture-mode cupti `
  --collector build/compat-audit-collector/Release/xvram_compat_audit_collector.dll `
  --output-dir artifacts/new-14b-ub128 --no-text
python -m xvram.compat_audit_observations `
  --trace artifacts/new-14b-ub128/cupti-trace.jsonl `
  --json artifacts/new-14b-ub128/observations.json
```

Repeat the capture for both models and microbatch 1/128. The observations command
is offline, accepts 1–16 traces, and refuses to overwrite its output. It emits
`xvram.cuda_compat_audit_observations` v1, validated by
[`cuda-compat-audit-observations-v1.schema.json`](../schemas/cuda-compat-audit-observations-v1.schema.json).
Trace v2 is validated by its [separate schema](../schemas/cuda-compat-audit-trace-v2.schema.json).
The original audit report remains v1. See the source-index guide for fetching the
small hash-pinned source dependencies and indexing these four `report.json` files.

## Recorded RTX 3070 results — 2026-09-06

Four new CUPTI runs of the unchanged official application/backend completed with
exit zero and drained their owned process trees. The profile still uses eight GPU
layers: **these are native CPU-offload observations, not xVRAM oversubscription**.
Visible output digests match the previous per-model observations; that is not a
tensor-level or numerical-equivalence proof. No new speed comparison is claimed.

| Case | GPU activities, all reconciled | Runtime → Driver → GPU | Direct Driver | Resolver calls / non-null results |
| --- | ---: | ---: | ---: | ---: |
| 14B, microbatch 1 | 26,860 | 26,860 | 0 | 765 / 764 |
| 14B, microbatch 128 | 8,776 | 8,680 | 96 | 773 / 772 |
| 32B, microbatch 1 | 26,758 | 26,758 | 0 | 765 / 764 |
| 32B, microbatch 128 | 9,868 | 9,740 | 128 | 773 / 772 |

All **72,262 observed GPU activities** reconcile without correlation issues.
The 3,076 resolver calls describe callback layers, not distinct entry points or
proven application invocation routes. Each capture contains one successful API
return for `cuDeviceGetNvSciSyncAttributes` without a non-null result; the observer
retains it as unresolved, without guessing whether it is an unused optional path.
All four traces still have incomplete terminal evidence and unknown API semantics.
They cannot justify complete coverage or GO.

The [acceptance record](acceptance/phase6b0-refinement-20260906.json) contains source,
collector and artifact hashes, counts and validation results. Full local artifacts
are under `artifacts/phase6b0-refinement-20260906`; original Phase 6b.0 captures were
not overwritten. Local checks passed: 84/84 Release CTest, 80/80 serial Debug CTest,
135 focused Python tests (two POSIX-only skips), warning-clean collector/core and
installed-package smoke. An initial parallel Debug run during 32B profiling hit two
existing controller timeouts; the full serial rerun passed without changing code
or deadlines. CI also exercises both offline modules and the installed source manifest.

### CI test organization follow-up

The first Windows CI attempt exposed duplicated audit discovery in the general
Python suite and a final-exit-grace test that charged Python startup against a
0.5-second retirement watchdog. The unchanged commit passed all seven jobs on
retry; the approved test-only correction removes both sources of variability.
The general suite now selects 114 tests and the audit suite selects 135, with
their existing 60/90-second CTest limits unchanged. A separate selection contract
guards the disjoint, exhaustive flat-module partition, including legacy names
and future modules, and rejects unsupported discovery layouts/hooks explicitly.
The grace arithmetic check uses real protocol frames with a modelled process and
controller-local clock; live-child success, crash, hang and reap tests remain.
Production runtime, watchdog settings and frozen contracts are unchanged.
Post-fix local CTest passed 85/85 Release and 81/81 serial Debug from the Visual
Studio developer environment; the selection contract's seven cases also passed.

## Remaining decision gates

Resolver observations do not prove actual invocation routing or owning module.
Source candidates do not prove compiled argument ABI, tensor/alias/indirect ranges,
native scratch consumption or admissible working sets. Native arena lifetimes,
device ordering and live WDDM admission still need a verified model. Stock-process
CUPTI finalization and the existing Nsight driver-support warning remain unresolved.

The next design must establish those concrete contracts while retaining the original
backend, or explicitly ask to change the boundary. This refinement does not authorize
exit hooks, backend rebuilds, private cuBLAS ABI guesses, kernel replacement or
Phase 6b.1 implementation. It is evidence progress, not a transparency claim.
