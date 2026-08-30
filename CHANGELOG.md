# Changelog

All notable changes will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project intends to use semantic versioning once its public API is established.

## [Unreleased]

### Added

- Initial Windows-first architecture and Phase 0 capability-probe scaffold.
- Versioned schema v2 with a calibrated, correctness-checked H2D/D2H copy-and-compute
  overlap benchmark backed by a hash-identified embedded PTX module.
- Explicit VMM oversubscription proof with stable virtual addresses, reusable physical
  handles, event-safe remapping, controller isolation, and `xvram.vmm_poc` schema v1.
- Event-safe multi-allocation residency cache with bounded staging, CLOCK/LRU policies,
  prefetch, dirty write-back, live CUDA/WDDM targets, `xvram-cache-bench`, and strict
  report/trace schemas.
- Experimental 64-bit C ABI v1 exposed only through `xvram_get_api`, including opaque
  sessions and allocations, isolated/attach-current contexts, explicit transactions,
  prefetch, asynchronous operations, allocation priority hints, and telemetry.
- Tiled FP16/BF16/FP32-TF32/FP64 GEMM planning with exact strided working sets, bounded
  workspace, dynamic cuBLAS/cuBLASLt dispatch, and compatibility fallback.
- `xvram-gemm-bench`, the `XVG1` isolated-worker protocol, strict
  `xvram.gemm_bench` report schema v1, SDK install/export support, and Phase 3 CI and
  RTX 3070 acceptance coverage.
