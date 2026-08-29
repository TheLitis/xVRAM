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

## Phase 1: explicit VMM proof of concept

- Reserve a logical address range larger than VRAM.
- Reuse physical VRAM chunks under stable virtual addresses.
- Remap only at CUDA event boundaries.
- Process a sequential logical array larger than VRAM with a tiled kernel.
- Detect corruption, OOM, timeout, and remap overhead.

## Phase 2: residency cache

- Chunk table and conservative clean/dirty state machine.
- Bounded pinned staging pool.
- H2D/D2H streams, event ownership, and prefetch queue.
- WDDM-aware dynamic target and pluggable eviction policy.
- Trace and metrics pipeline.

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
goal, not a Phase 1 promise.
