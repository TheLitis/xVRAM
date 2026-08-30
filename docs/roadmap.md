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

## Phase 2: event-safe residency cache — implementation complete, hardware gate pending

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

The implementation and no-GPU contract gates are complete. The remaining exit criterion
is the reproducible RTX 3070 gate: suite runs at `0.8x`, `1.1x`, `1.5x`, and `2.0x` total
VRAM for CLOCK and LRU, followed by the `1.5x` CLOCK budget-pressure run. The checked-in
`scripts/run-phase2-rtx3070-acceptance.ps1` command validates all nine reports, compares
policy digests, enforces scenario-specific cache accounting, and checks that no worker
remains.

## Phase 3: known operations

- Tiled cuBLAS/cuBLASLt GEMM working-set planner.
- Workspace declarations and reusable hot allocations.
- Explicit operation transactions through the stable C ABI.

## Phase 4: PyTorch

- Pluggable allocator/MemPool integration.
- Tensor classification and lifetime tracking.
- FX/compile graph next-use hints.
- Inference, then backward and optimizer-state scheduling.

## Phase 5: adaptive compression

- Per-chunk compression history and cost model.
- Raw, CPU-encode/GPU-decode, and GPU-codec paths.
- Compression is selected only when predicted end-to-end time is lower.

## Phase 6+: interception and instrumentation

- Conservative CUDA Runtime and Driver API interception.
- Known cuBLAS/cuDNN launch semantics.
- Access-range profiling, PTX indirection, and automatic tiling research.

Transparent support for an arbitrary unchanged CUDA executable is a long-term research
goal, not a Phase 1 or Phase 2 promise.
