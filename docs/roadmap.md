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

### Phase 4b: lease-scoped static inference — complete

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

## Phase 5: adaptive lossless compressed backing — complete

- Production `xvram_residency` owns an authoritative `HostBackingStore`; the frozen
  Phase 2 `CacheManager` remains the raw regression implementation.
- Each logical chunk has one immutable-generation host representation: `invalid`,
  `implicit_zero`, `raw`, or `xvram_lz4_blocks_v1`. The container uses independent
  64 KiB LZ4 blocks inside the default 64 MiB residency chunk, preserves the valid tail,
  and stores an individual block raw rather than expanding it.
- Host writes create and verify a replacement generation before atomic commit. Full
  writes do not read the prior generation; partial writes materialize only affected
  blocks. Proven-dead data can be discarded after its event retires, while ordinary
  release retains write-back semantics.
- A write-capable lease reserves a full raw spill credit for each first-dirty chunk
  before launch. An admission failure returns host OOM before data changes; codec failure
  can fall back only while a valid authority still exists.
- Raw, CPU LZ4 encode/GPU decode, and nvCOMP GPU codec paths share the production
  residency state machine. Codec slots, event generations, encode/decode streams, and
  workspace are independently bounded and cannot be reused before successful event
  retirement and atomic host commit.
- The adaptive EWMA model accounts for compression ratio, CPU/GPU encode/decode,
  transfer, staging, CPU queueing, reuse, dirty rate, and SM opportunity cost. After two
  calibration samples it requires a predicted win of at least `max(10%, 50 us)`; three
  unfavorable probes suppress compression for that content generation. Capacity mode
  may choose a smaller lossless representation even when raw is faster.
- There is no fixed logical multiplier. Automatic sizing stays at the conservative
  raw-safe `1.5x VRAM`; explicit sizes are constrained by CUDA VA, current host-store
  budget, working-set fit, and live CUDA/WDDM budget. Compressible workloads may exceed
  `2x`; incompressible workloads gain no artificial capacity.
- Public `<xvram/xvram_v2.h>` embeds the frozen v1 table and adds explicit compression
  configuration and telemetry without adding an export. ABI v1 remains eager-raw and
  never loads a codec. The private PyTorch table follows the same v2 rule; Python keeps
  `compression="off"` unless the caller explicitly requests `adaptive` or `capacity`.
- `xvram-compression-bench` uses the isolated `XVZ1` protocol, strict
  `xvram.adaptive_compression` v1 reports, and optional `xvram.compression_trace` v1
  JSONL. Timeout, malformed protocol, trace failure, cleanup failure, and no-driver
  paths retain the existing exit-code family and controller-owned output.
- LZ4 1.10.0 and nvCOMP 5.3.0.16 are hash-pinned. nvCOMP is staged and installed
  app-locally, loaded dynamically, and tested after relocation. The Ampere gate uses the
  CUDA codec backend; the Blackwell-only hardware Decompression Engine is out of scope.

The local gates passed on 2026-09-05 at clean Release commit
`5220b86114db2750729f4a0ea3aa8c12bdf13b62`: 69/69 no-driver/contract tests, all 12 core
RTX 3070/WDDM runs, the public SDK v2 hardware smoke, and all 11 PyTorch runs (six
compression-off Phase 4b regressions plus five v2 cases). Every hardware report exited
zero, passed schema/trace/semantic validation, matched its reference, completed cleanup,
and left no worker. All eleven inference outputs matched the reference digest exactly.

The core matrix demonstrated forced-path and CLOCK/LRU parity, adaptive raw selection
for incompressible data, GPU encode-before-D2H, and event-safe budget shrink/grow. The
`4x` stress case was executed, not skipped: 34,357,641,216 logical bytes fit in 148,364,448
stored payload bytes under a 28,923,184,128-byte host cap. This highly compressible
fixture proves elastic admission, not a typical-model ratio or a product maximum.
The controlled `3x` scan median was 6,676.4934 ms raw versus 6,590.6049 ms GPU-LZ4
(1.286% lower); no general workload speedup is claimed.

PyTorch v2 passed the 42-layer incompressible adaptive case, all three 47-layer
structured capacity cases (CLOCK/LRU and prefetch 0/2), and sequence-128 attention smoke.
The three 47-layer graph hashes and output digests matched; weight D2H remained zero.
See the [acceptance evidence](adaptive-compression.md#recorded-local-results) for exact
artifact locations and accounting. The full Windows/Linux Debug/Release, Clang
ASan/UBSan, installed-package/relocation, and PyTorch Stable-ABI 2.11→2.13 matrix passed
at commit `97dde36184ebd25dc70553606aba5a468dedb0d6` in
[GitHub Actions run 33928261245](https://github.com/TheLitis/xVRAM/actions/runs/33928261245).
This completes Phase 5 within its stated scope; version remains `0.1.0-dev`.

## Phase 6a: explicit synchronous CUDA/cuBLAS adapter — in progress

- Optional host-only C++ facade and size-tagged C control table, with one isolated worker.
- Stable pointer registry and VA tombstones; production raw residency and shared tiled SGEMM.
- Positive column-major FP32 N/T GEMM with padding/interior pointers and strict FP32 math.
- Isolated benchmark, bounded XVI1 protocol, strict report/trace contracts, and no-driver tests.
- Default-OFF regression, ON builds, negative/installed consumers, and app-local cuBLAS checks.

Implementation and validation are in progress. Completion requires the recorded RTX 3070
NN/NT/TN/TT, split-K, and 1.1x/1.5x/2.0x CLOCK/LRU gate plus the complete Actions matrix.
See the [compatibility guide](cuda-compat.md). This phase requires an application rebuild;
it does not intercept unchanged executables or support arbitrary kernels/streams.

## Phase 6b+: interception and instrumentation

- Conservative CUDA Runtime and Driver API interception.
- Known cuBLAS/cuDNN launch semantics.
- Access-range profiling, PTX indirection, and automatic tiling research.

Transparent support for an arbitrary unchanged CUDA executable is a long-term research
goal, not a Phase 1, Phase 2, Phase 3, or Phase 4b promise.
