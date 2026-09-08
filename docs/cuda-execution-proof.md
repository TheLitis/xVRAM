# Phase 6b.0g — execution proof obligations (in progress)

## Implemented result, not completion of the four gates

This first proof-stage increment adds a bounded, no-driver verifier and an offline
cross-evidence report. It does **not** yet prove live memory bounds, launch-to-cubin
identity, application device ordering or terminal capture. No new native hooks,
kernel-argument reads, GPU submissions or residency operations were introduced.

The new report recomputes the accepted Phase 6b.0f metadata/census traces and compares
live parameter offsets/sizes with the pinned static binary evidence:

| Evidence | Result |
| --- | ---: |
| Observed GPU launches / metadata calls | 8,776 / 8,776 |
| Live kernel-name/layout groups | 39 |
| Groups with exactly one identical static ABI candidate | 35 |
| Groups with an ABI mismatch / ambiguous matching candidates | 0 / 0 |
| Groups absent from the backend static inventory | 4 |
| Launches belonging to those four library groups | 192 |
| Proven live cubin bindings / memory-bound groups | 0 / 0 |

The four missing groups are two CUTLASS kernels with 360-byte parameter structures
and two Ampere GEMM kernels with 184-byte structures. Each has 48 observed calls.
They need a supported public library-operation contract; their opaque members are
not decoded by guessing offsets. Static candidates are separate-capture evidence,
not native module identities. Even one exact name-and-ABI match stays a candidate.

All four gates in `xvram.cuda_execution_evidence` v1 remain explicitly unproven and
the decision remains `NO-GO`. Exit zero means that this evidence analysis completed.
The input runs remain ordinary eight-layer CPU-offload diagnostics, not xVRAM
oversubscription. This increment reuses those immutable captures; it ran no new
GPU workload and makes no throughput or tensor-equality claim.

## Conditional checking core

`compat_audit_proof` checks implications of **declared witness facts**. It is not yet
wired to a live collector and does not authenticate those facts. Synthetic model
success must never promote a hardware report to GO.

- Memory access must fit the declared tensor subrange, which must fit its exact
  allocation generation. Fitting a larger allocation alone is insufficient.
- Checked 64-bit arithmetic, nonzero base/chunk alignment, tail chunks and the
  existing chunk-union model bound the declared footprint.
- Overlapping read aliases are allowed; overlapping accesses involving a write
  are rejected by this conservative profile. Stride-hole envelopes can overreject.
  Separate allocation IDs still require independent non-aliasing proof.
- A single-context, explicit-stream vector-clock model derives dependency edges
  from stream submission and event record/wait, never timestamps or host API exit.
- Each record generation snapshots dependencies. Re-recording an event cannot
  retroactively extend an already submitted wait. Stale event queries and ABA
  generations are rejected; successful retirement includes transitive users.
- Every declared mapping user must retire before reuse. `not_ready` proves no
  completion; any other query error poisons the model.
- Closing submission admission still permits event drain. Conditional terminal
  closure requires producer quiescence, closed API pairs, retired GPU nodes,
  completed flush, returned buffers, consumed records and zero drops/errors.
  An attempted later submission invalidates the closure.

Limits are 100,000 state items, 64 stream identities and one million stored clock
entries. Range checking is bounded and uses an O(n log n) alias sweep. Default-stream
implicit semantics, graphs, PDL, cross-context waits, in-process unload and unknown
indirect accesses are outside this model, not silently treated as ordinary edges.
The [CUDA event contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
is the reference for record/query semantics.

## Evidence verification and reproducibility

The analyzer rejects incomplete/failed metadata input, a different backend hash,
modified reports, trace/report disagreement, duplicate JSON keys, invalid numbers,
input growth and changed bytes during analysis. Reports use normalized names,
relative metadata, local indices and hashes, not pointers or native handles. The
strict schema and semantic validator reject promoted proof flags, extra fields,
missing blockers, duplicate identities and contradictory counters.

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
python -m xvram.compat_audit_execution `
  --probe-report artifacts/phase6b0f-legacy-20260908/accepted-metadata/probe.json `
  --probe-trace artifacts/phase6b0f-legacy-20260908/accepted-metadata/launches.jsonl `
  --census-report artifacts/phase6b0f-legacy-20260908/accepted-metadata/census-report.json `
  --census-trace artifacts/phase6b0f-legacy-20260908/accepted-metadata/census.jsonl `
  --binary-evidence artifacts/phase6b0-binary-evidence-20260906/binary-evidence.json `
  --json artifacts/new-execution-evidence.json
```

Use a fresh output path. `scripts/validate-execution-evidence.py` accepts those five
input options plus `--report` and recomputes the result in addition to validating
schema and semantics. The module CLI ships in the installed Python package and
requires neither CUDA nor PyTorch. Existing audit discovery includes its tests in
Windows/Linux CI. All previous ABI/schema/export bytes remain unchanged; the two
Phase 6b.0f v2 trace hashes are additionally frozen by tests.

## Next native witness work — not implemented by this increment

1. **Identity:** observe successful `cuLibraryLoadData` / `cuLibraryGetKernel` and
   unload lifetimes using generation-scoped local IDs. These API labels are present
   in the accepted census, but their arguments/outputs were not captured. The
   documented [`cuKernelGetLibrary`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html)
   can establish a kernel-to-library edge; it does not by itself prove loaded cubin
   bytes or native-to-CUPTI module equality. Loaded input/image provenance, selected
   cubin identity and lifecycle correlation still need explicit validation. Do not
   infer them from nesting, name uniqueness or equal-looking numeric IDs.
2. **Memory:** after identity closure, capture typed scalar inputs and normalized
   tensor/allocation generations for a bounded kernel family (ordinary q8 quantize
   is a source-model starting point). Validate indirect descriptors, source/binary
   assumptions, alias exclusions and native scratch before invoking the range model.
3. **Ordering:** capture actual context/stream/event generations, record/wait/query
   outcomes and complete mapping-user sets. Feed those facts into the conditional
   model only after coverage is established; extend unsupported execution semantics
   explicitly or reject them.
4. **Completeness:** establish a producer-quiescence boundary while the application
   and CUDA context are still valid, then GPU drain and final CUPTI delivery outside
   callbacks and loader lock. A process-exit snapshot, no reported loss or no
   outstanding buffers alone cannot satisfy this gate. Keep watchdog/reap behavior.

Only closure of these native witness paths, including required 32B coverage, can
complete the proof stage. The code in this increment is the checking foundation,
not that closure. Production memory behavior and version `0.1.0-dev` are unchanged.

The [acceptance record](acceptance/phase6b0g-proof-20260908.json) pins the checker
source, prior inputs and new offline report, and records local validation results.

Local validation: 86/86 Release CTest, 82/82 Debug CTest, 350 audit Python tests
(four platform skips), including 23 conditional-proof and 13 cross-evidence tests.
The accepted report passed schema/semantic/exact-input recomputation. Installed
Python smoke and all prior frozen-contract checks passed.
