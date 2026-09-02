# Roadmap

## Phase 0: capability and transport probe — complete

- Runtime-load the CUDA Driver API.
- Enumerate devices and relevant attributes.
- Query device/host/NUMA VMM granularities only when capability-gated.
- Match CUDA LUIDs to DXGI adapters and report process memory budgets.
- Measure pinned H2D, D2H, and full-duplex transport safely.
- Measure copy/compute overlap with a versioned, hash-identified synthetic PTX module.
- Emit human-readable and versioned JSON reports.

Exit criterion reached: the RTX 3070 development machine produces a complete, repeatable
report, the synthetic compute and transfer outputs are verified, overlap is measured in
both directions, and the no-driver path is covered in Windows/Linux CI.

## Phase 1: explicit VMM proof of concept — complete

- Reserve a logical address range larger than VRAM.
- Reuse physical VRAM chunks under stable virtual addresses.
- Remap only at CUDA event boundaries.
- Process a sequential logical array larger than VRAM with a tiled kernel.
- Detect corruption, OOM, timeout, and remap overhead.

Exit criterion reached: the RTX 3070 development machine completed three independent
12 GiB runs and one 16 GiB stress run with two passes in both reference and pipeline
modes. Every report validated against `xvram.vmm_poc` schema v1, every GPU result matched
the CPU reference, handle reuse and stable addresses were observed, unsafe remaps and
mismatches remained zero, all cleanup flags were true, diagnostics were empty, and no
worker process remained. Windows/Linux no-driver behavior, injected CUDA faults, worker
crash/hang/protocol failures, and the report fixtures are covered in CI.

## Phase 2: event-safe residency cache — complete

- Multi-allocation logical heap with stable padded VA reservations and pageable backing.
- Conservative state machine with generation-safe events and explicit pin/frame/staging
  ownership.
- Independent bounded H2D and D2H staging pools, dirty write-back, demand loading, and
  speculative prefetch.
- Cost-aware CLOCK and deterministic LRU policies with a manager-side victim recheck.
- CUDA/WDDM-aware dynamic target, immediate shrink, hysteretic grow, and bounded OOM
  recovery.
- Isolated `xvram-cache-bench` controller/worker, strict report and JSONL trace schemas,
  deterministic CPU verification, fake-CUDA fault coverage, and no-driver CI.

The reproducible RTX 3070 gate completed on 2026-08-30 against 8,589,410,304 bytes of
physical VRAM. CLOCK and LRU each completed the five-scenario suite at `0.8x`, `1.1x`,
`1.5x`, and `2.0x`, producing eight schema-valid reports and 40 accepted scenario cases.
The largest logical backing was 17,178,820,608 bytes. Policy digests matched at every
size, read-only D2H remained zero, reuse hit rate remained above 99.7%, and oversubscribed
runs demonstrated real eviction, dirty write-back, and handle reuse.

The separate `1.5x` CLOCK budget-pressure run observed four target shrinks followed by
two hysteretic grows. Across all nine reports, mismatch, unsafe transition, and unsafe
remap counts remained zero; cleanup ledgers were complete, diagnostics were empty, and
no worker process remained. The checked-in
`scripts/run-phase2-rtx3070-acceptance.ps1` command reproduces the gate and validates the
strict schemas, policy digests, semantic accounting, cleanup, and process isolation.

## Phase 3: public SDK and known GEMM — complete

- Size-tagged C ABI v1 behind the single `xvram_get_api` export, with opaque session,
  allocation, plan, and asynchronous-operation handles.
- Isolated and attach-current CUDA context modes serialized through a per-session worker
  thread.
- Explicit access-range transactions, bounded workspace declarations, prefetch,
  allocation priority hints, asynchronous poll/wait/cancel, and synchronous wrappers.
- M/N/K tiled ordinary GEMM for FP16, BF16, FP32/TF32, and FP64 with row/column-major and
  N/T operands.
- Dynamic cuBLAS/cuBLASLt dispatch, cuBLASLt heuristic caching, and cuBLAS fallback.
- Isolated `xvram-gemm-bench`, `XVG1` worker protocol, strict
  `xvram.gemm_bench` v1 reporting, no-driver tests, and installable CMake package.

The reproducible RTX 3070 gate completed on 2026-08-31 against 8,589,410,304 bytes of
physical VRAM. All nine schema-valid reports exited zero: the mixed suite; FP16, BF16,
strict FP32, TF32, FP64 transpose, and explicit FP32 K-panel boundary cases; plus `1.1x`
and `1.5x` full-validation oversubscribed cases. Every numerical digest matched, mismatch,
unsafe-remap, and unsafe-transition counts remained zero, cleanup was complete,
diagnostics were empty, and no worker process remained.

The `1.1x` case validated 9,447,689,772 logical bytes and 787,307,481 output elements,
with 527 evictions/handle reuses and 47 dirty write-backs. The `1.5x` case validated
12,882,542,700 logical bytes and 1,073,545,225 output elements, with 7,211
evictions/handle reuses and 64 dirty write-backs. Both oversized reports set every proof
flag true while keeping their cache and workspace within the observed live target. The
checked-in `scripts/run-phase3-rtx3070-acceptance.ps1` command reproduces the complete
format, planner, residency, correctness, cleanup, and process-isolation gate.

## Phase 4: PyTorch — complete

### Phase 4a: resident allocator boundary — complete

- Pluggable allocator/MemPool integration through stable, fully resident VMM segments.
- Explicit tensor classification and conservative lifetime metadata.
- Static FX graph next-use, release-candidate, and prefetch-candidate hints.

The Phase 4a boundary has been validated on Windows with PyTorch 2.13/CUDA 13.0 and an
RTX 3070: native segment callbacks, stable mappings, `SetAccess`, event-fenced teardown,
pool caching, and cross-stream `record_stream` retirement complete without quarantine.
This boundary does not oversubscribe PyTorch tensors. The allocator API cannot see
operator access ranges, so live tensor mappings remain resident and immutable until
PyTorch releases their complete backing segment.

### Phase 4b: lease-scoped static inference — implementation and local gate complete

- Two-phase external leases over the Phase 2 pageable residency runtime, with one
  runtime-owned compute stream, monotonically generated events, live-view accounting,
  event-safe dirty retirement, prefetch, and proven-dead activation discard.
- Private, size-tagged native control table and Stable-ABI
  `xvram_internal::_wrap_resolved_v1` bridge; the public Phase 3 and Phase 4a ABIs remain
  unchanged.
- Strict `torch.export` planner for static shapes, aliases, tied weights, activation-slot
  coloring, embedding ranges, bounded scratch, and a deliberately limited operator
  allowlist with no eager fallback.
- `xvram.torch.InferenceRuntime`, a `torch.compile` backend adapter strictly validated
  against the canonical export plan, CPU and chunked/meta state providers, CPU output by
  default, and explicit bounded CUDA materialization.
- Private synchronous tiled-GEMM route for oversized `mm` and bias-free `linear`, using
  the shared Phase 3 planner/executor and event-safe residency operations. Other
  oversized operators remain unsupported.
- Isolated `xvram-torch-bench` controller/worker, bounded `XVT1` protocol, strict
  `xvram.pytorch_inference` v1 report, optional `xvram.pytorch_trace` v1 JSONL trace,
  controller watchdog, schema/semantic checks, and worker reaping.
- Windows/Linux Stable-ABI build/load coverage with pinned CPU PyTorch 2.11, plus native,
  planner, fake-backend, protocol, report, and no-driver test targets.

The local Phase 4b hardware criterion completed on 2026-09-02. All six RTX 3070
scenarios exited zero and passed schema, JSONL trace, and semantic validation: CLOCK at
16, 23, and 31 layers; LRU at 31 layers; CLOCK at 31 layers with prefetch distance zero;
and the two-layer sequence-128 attention smoke. The three 31-layer output digests agree.
The distance-zero report recorded zero submitted/retired prefetches, while both
distance-two 31-layer reports recorded and retired non-zero prefetch work.

Every oversubscribed report demonstrated eviction and frame reuse; event, view,
mapping, cache, trace, and worker-isolation accounting reconciled; cleanup was complete;
and diagnostics were empty. The implementation and local hardware gate are therefore
complete. The full Windows/Linux Debug/Release, Clang ASan/UBSan, installed-package,
and PyTorch Stable-ABI 2.11→2.13 matrix passed in
[GitHub Actions run 33676882740](https://github.com/TheLitis/xVRAM/actions/runs/33676882740),
completing Phase 4b within its stated scope.

Phase 4b remains limited to static-shape, single-GPU, forward inference. Autograd,
backward/optimizer scheduling, CUDA Graphs, dynamic control flow, distributed/NCCL,
arbitrary extensions, and transparent execution of unsupported oversized operators are
future work.

## Phase 5: adaptive compression

- Per-chunk compression history and cost model.
- Raw, CPU-encode/GPU-decode, and GPU-codec paths.
- Compression is selected only when predicted end-to-end time is lower.

## Phase 6+: interception and instrumentation

- Conservative CUDA Runtime and Driver API interception.
- Known cuBLAS/cuDNN launch semantics.
- Access-range profiling, PTX indirection, and automatic tiling research.

Transparent support for an arbitrary unchanged CUDA executable is a long-term research
goal, not a Phase 1, Phase 2, Phase 3, or Phase 4b promise.
