# xVRAM

**xVRAM** is an experimental, Windows-first runtime for software-managed tiered CUDA
memory. System RAM is the large capacity and backing tier; physical VRAM is the fast,
write-back residency tier where kernels execute.

The project is aimed at workloads whose total data is larger than VRAM but whose
per-operation working set can be made resident. The runtime combines stable CUDA
virtual addresses, asynchronous transfers, explicit working-set declarations,
WDDM-aware budgeting, cache policy, and library-aware tiling.

> [!WARNING]
> xVRAM now exposes an experimental, versioned C ABI, tiled GEMM integration, and a
> resident-only PyTorch MemPool adapter. A separate lease-scoped PyTorch inference
> runtime is implemented and has passed its local six-scenario RTX 3070 gate and the
> complete Windows/Linux GitHub Actions matrix. Phase 5 adds opt-in, lossless compressed
> host backing and has passed both RTX 3070 gates and the complete Windows/Linux CI matrix.
> The SDK remains pre-release, does not transparently extend an arbitrary application's
> VRAM, and must not be used for production workloads.

## Memory model

```text
CUDA application / framework adapter
                  |
        logical extended GPU heap
                  |
      residency and transfer engine
            /             \
   VRAM resident cache    system RAM backing store
            |
        GPU kernels
```

The strict execution invariant is:

```text
WorkingSet(kernel) <= available physical VRAM budget
```

There is no transparent, recoverable GPU page-fault path in the initial architecture.
Every range a kernel may touch must be made resident before launch. Operations larger
than the resident budget therefore require library-aware tiling, graph transformation,
or later PTX instrumentation.

## Phase 0 foundation

`xvram-probe` discovers the facts that later policy must use instead of hard-coding a
specific GPU or driver:

- CUDA driver and device identity;
- device, host, and host-NUMA VMM capabilities;
- the closest host NUMA node used for host-NUMA allocation probes;
- CUDA memory-pool and mapped/pageable-host-memory capabilities;
- allocation granularities for supported VMM locations;
- free and total CUDA memory;
- copy-engine and architectural limits;
- an active CUDA VMM reserve/map/copy/unmap/remap/data-integrity smoke test;
- NVIDIA display-driver and WDDM/TCC/MCDM model observations through NVML;
- CUDA LUID to DXGI adapter matching on Windows;
- WDDM local/non-local process budget and usage;
- a calibrated copy/compute overlap benchmark using a hash-identified embedded PTX module;
- host memory and operating-system information.

The probe loads the system CUDA Driver API and NVML dynamically. Building does not
require a complete CUDA Toolkit: when compatible headers are absent, CMake downloads
NVIDIA's platform-specific CUDA 13.3 redistributable header archives and verifies pinned
SHA-256 hashes. Running without an NVIDIA driver still produces a valid diagnostic
report rather than a loader crash.

## Phase 1 explicit VMM proof

`xvram-vmm-poc` processes a deterministic logical `uint32` array that is strictly
larger than total VRAM. The complete array remains in pageable RAM while one reference
slot or two pipelined VMM slots are repeatedly mapped at stable CUDA virtual addresses.
Only the bounded slots and their staging buffers are resident or pinned.

Every tile is checked against the CPU before write-back, both modes receive identical
input, and a second full-array verification produces matching 128-bit digests. Remapping
is permitted only after a successful completion-event query. A controller isolates the
CUDA worker, enforces progress and total deadlines, and emits a strict
`xvram.vmm_poc` v1 report even when it must terminate the worker.

This is a fixed FIFO proof, not the Phase 2 residency cache. See the
[VMM proof guide](docs/vmm-poc.md) for its safety model, CLI, report contract, and exit
codes.

## Phase 2 residency cache

`xvram-cache-bench` exercises the internal `xvram_residency` cache engine. Each logical
allocation owns pageable host backing and a stable padded CUDA VA reservation. Reusable
VMM frames form a bounded write-back cache, while independent H2D and D2H pinned staging
pools remain bounded by `--staging-slots`.

The benchmark covers sequential, hot-set reuse, deterministic random, read-only, and
write-heavy access traces. Cost-aware CLOCK is the default policy; deterministic LRU is
available as a baseline, and `--policy both` runs both from identical backing and seeds.
Every operation is checked against the CPU reference. Dirty frames are written back
before reuse, mappings are removed only after their event generation completes, and
live CUDA/WDDM observations can shrink or cautiously grow the cache target.

The public process owns output and supervises an isolated worker through the versioned
`XVC1` protocol. It emits a strict `xvram.residency_cache` v1 report and can optionally
write an `xvram.residency_trace` v1 JSONL trace. See the
[residency-cache guide](docs/residency-cache.md) for the CLI, state machine, report and
trace contracts, safety rules, and hardware acceptance procedure.

## Phase 3 SDK and tiled GEMM

The `xvram` shared library exposes ABI v1 through the single exported symbol
`xvram_get_api`. Its opaque sessions and allocations retain the Phase 2 pageable-backing,
stable-VA, event-generation, write-back, staging, policy, and live-budget rules. The ABI
supports isolated CUDA contexts by default and opt-in attachment to the caller's current
context, asynchronous operations with polling/waiting, synchronous wrappers, prefetch,
explicit working-set transactions, and allocation priority hints.

Generic transaction callbacks run on the session worker thread with the session CUDA
context current. A callback may enqueue work only on the supplied stream; it may not
synchronize or replace the context, retain resolved device addresses, or use them after
return. xVRAM records the completion event after the callback and does not make any
declared range evictable until that event retires.

The first known operation is ordinary real-valued GEMM. Its planner divides M, N, and K
so each A/B/C tile working set and the bounded library workspace fit the live cache
target. cuBLASLt is preferred and cached per tile signature, with cuBLAS GEMM as the
compatibility fallback. Strict FP16/BF16-output proofs keep the complete K dimension in
one tile and use the pedantic cuBLAS path with reduced-precision reduction disabled, so
intermediate values are never rounded through low-precision C storage. FP16, BF16,
FP32/TF32, and FP64, row/column-major layouts, and N/T operands are represented by ABI
v1. Batching, complex values, fused epilogues, framework adapters, and transparent CUDA
interception remain out of scope.

`xvram-gemm-bench` is the isolated controller/worker hardware gate for this path. It
validates every pass of tiled numerical output, reports algorithms, workspace,
GEMM-only throughput, end-to-end residency timing, and residency telemetry, and writes
the strict `xvram.gemm_bench` v1 report without serializing raw CUDA virtual addresses.

## Phase 4a PyTorch resident allocator

The experimental `xvram_torch_allocator` plugin can back a selected
`torch.cuda.MemPool` with stable CUDA VMM segments. Each PyTorch caching-allocator
segment receives one complete physical mapping, `cuMemSetAccess`, and an event-fenced
full-range teardown. A size-tagged telemetry ABI reports callbacks, mapped padding,
cleanup reconciliation, capture rejection, and quarantine without exposing raw virtual
addresses.

The Python package adds conservative tensor-role classification and static FX next-use
hints. Those hints are advisory: `CUDAPluggableAllocator` does not expose operator access
ranges, so Phase 4a never evicts or remaps a live tensor and does not claim PyTorch
oversubscription. See the [PyTorch MemPool guide](docs/pytorch-mempool.md) for setup,
stream-lifetime rules, telemetry, tests, and exact limitations.

## Phase 4b lease-scoped PyTorch inference — complete

`xvram.torch.InferenceRuntime` strictly exports one static, single-GPU forward graph and
stores parameters, inputs, outputs, and planned activations in pageable xVRAM backing.
Before each ATen region, the residency runtime makes the declared chunks resident and
returns non-owning CUDA views only for the lifetime of an event-fenced compute lease.
Python never receives a CUDA virtual address, only one runtime-owned stream may submit
the region, and mappings cannot be removed or reused until the completion generation
retires.

The private `xvram_torch_runtime` bridge uses PyTorch's Stable ABI and attaches to the
current primary CUDA context. The strict planner rejects graph breaks, dynamic shapes,
RNG, hidden streams, unsupported mutation/aliasing, and operators outside its allowlist
before launch. `mm` and bias-free `linear` whose declared working set exceeds the live
cache target use the shared event-safe tiled GEMM executor; other oversized operators
are rejected instead of falling back to eager execution.

`xvram-torch-bench` isolates CUDA work behind the `XVT1` controller/worker protocol and
emits the strict `xvram.pytorch_inference` v1 report plus an optional
`xvram.pytorch_trace` v1 JSONL trace. The implementation, CI Stable-ABI build/load job,
and RTX 3070 acceptance script are present. See the
[lease-scoped inference guide](docs/pytorch-inference.md) for the supported graph,
safety invariants, CLI, proof semantics, current limitations, and delivery status.

Current status (2026-09-02): all six local RTX 3070 scenarios have produced exit-0
reports and passed schema, trace, and semantic checks—CLOCK 16/23/31, LRU 31, CLOCK 31
with prefetch disabled, and the two-layer sequence-128 attention smoke. The three
31-layer digests match; distance-zero recorded no prefetch while distance-two recorded
and retired real prefetch work. Cleanup is complete and diagnostics are empty. The
implementation and local hardware gate are complete. The full Windows/Linux build,
sanitizer, installed-package, and Stable-ABI 2.11→2.13 matrix passed in
[GitHub Actions run 33676882740](https://github.com/TheLitis/xVRAM/actions/runs/33676882740).

## Phase 5 adaptive lossless backing

Phase 5 replaces the production residency runtime's single flat host image with a
generation-safe `HostBackingStore`. Every logical chunk has exactly one authoritative
host representation: `invalid`, `implicit_zero`, `raw`, or the CPU-decodable
`xvram_lz4_blocks_v1` container. The outer residency chunk remains 64 MiB by default;
the container divides it into independent 64 KiB LZ4 blocks and stores an individual
block raw whenever compression would expand it.

The available paths are raw transfer, CPU LZ4 encode with GPU decode, and app-local
nvCOMP GPU encode/decode. `adaptive` selects a compressed path only after calibration
predicts an end-to-end win of at least `max(10%, 50 us)`. Explicit `capacity` mode may
retain a smaller compressed representation even when the raw path is faster. All paths
remain lossless, every candidate generation is verified before atomic commit, and a
write-capable lease must reserve enough raw spill capacity before GPU mutation begins.

There is no fixed `2x` logical-size ceiling. `auto` deliberately remains the raw-safe
`1.5x VRAM` default, while an explicit size is admitted from current CUDA VA, host-store,
working-set, VRAM, and WDDM budgets. Highly compressible data can therefore exceed `2x`;
incompressible data receives no fictitious capacity gain and can still fail preflight or
return host OOM without losing the last valid generation.

Public ABI v1 and the Phase 1--4 report contracts stay frozen. Opt-in SDK callers include
`<xvram/xvram_v2.h>` and request API v2 from the same sole `xvram_get_api` export. The
PyTorch frontend keeps `compression="off"` by default; `adaptive` and `capacity` require
the private v2 control table and never silently downgrade. `xvram-compression-bench`
isolates the core hardware proof behind `XVZ1`, writes `xvram.adaptive_compression` v1,
and optionally emits `xvram.compression_trace` v1 JSONL. See the
[adaptive-compression guide](docs/adaptive-compression.md) for the backing format,
budget model, CLI, safety rules, and recorded RTX gate results.

## Phase 6a explicit CUDA/cuBLAS integration

The optional `xVRAM::cuda_compat` target provides a synchronous host-only C++ facade for
selected CUDA memory/device calls and column-major FP32 SGEMM. Applications must rebuild
and initialize explicitly; xVRAM derives tiled residency ranges without caller-written
transactions. Stable logical pointers, retained VA tombstones, and the existing event-safe
runtime preserve the memory lifetime boundary. Compression is disabled in this profile.

Enable `XVRAM_BUILD_CUDA_COMPAT=ON` and use the official NVIDIA headers. Unsupported
native calls do not silently fall back. This is not unchanged-executable interception,
arbitrary kernel support, or a replacement for CUDA Runtime. See the
[compatibility guide](docs/cuda-compat.md) for integration, limits, benchmark, and gate status.

## Phase 6b.0 transparent-compatibility audit

`xvram-compat-audit` inventories binaries, captures diagnostic observations, and analyzes
whether a pinned, unchanged llama.cpp CUDA profile has a sufficiently proven memory and
execution contract for a future interceptor. The initial profile targets real Qwen2.5
14B and 32B Q4_K_M CLI inference, including prefill, decode, and KV-cache lifetimes.

This is a diagnostic tool, not transparent execution support. It does not change the
production residency runtime, remap application memory, replace kernels, or make a CPU
offload baseline into an xVRAM oversubscription proof. Unknown tensor ranges, missing
kernel contracts, dropped observations, or incomplete teardown require `NO-GO`.

The Python inventory/analyzer and no-driver tests do not require CUDA or PyTorch.
An optional observation-only CUPTI collector is enabled separately with
`XVRAM_BUILD_COMPAT_AUDIT_COLLECTOR=ON`; it is off in normal SDK builds. See the
[compatibility audit guide](docs/cuda-compat-audit.md) for the pinned profile, evidence
boundary, capture requirements, privacy, and reproduction workflow.

## Build

Requirements:

- 64-bit Windows 10/11 or a 64-bit Linux development host;
- CMake 3.25 or newer;
- a C++20 compiler (MSVC 2022/2026, Clang, or GCC).
- Python 3 plus `tests/requirements.txt` when running the JSON Schema contract test.

The SDK loads the CUDA Driver API and cuBLAS/cuBLASLt at runtime. A CUDA Toolkit is not
required to use an already-built SDK, but a compatible NVIDIA driver and cuBLAS runtime
must be discoverable for GPU work. Package builds may stage the hash-pinned official
CUDA 13.3 cuBLAS redistributable next to the SDK.

Compression builds use hash-pinned LZ4 1.10.0 sources and the official nvCOMP
5.3.0.16 CUDA 13 redistributable. The nvCOMP libraries and their license/notice are
staged beside compression-enabled binaries and installed app-locally; the runtime loads
them dynamically. Both configure and runtime loading verify the actual GPU library's
SHA-256 against the platform pin; runtime loading also checks the exact 5.3.0 semantic
version. Reports carry the verified file hash, which identifies package build 5.3.0.16.
nvCOMP is optional at execution time when compression is off or a CPU/raw fallback
remains valid. The RTX 3070 path uses nvCOMP's CUDA backend; it does not require the
Blackwell-only hardware Decompression Engine.

Install the test-only Python dependency:

```text
python -m pip install --requirement tests/requirements.txt
```

On Windows from a Visual Studio developer shell:

```powershell
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

You can also select a Visual Studio generator explicitly:

```powershell
cmake -S . -B build/vs -A x64
cmake --build build/vs --config Debug
ctest --test-dir build/vs -C Debug --output-on-failure
```

Run the probe:

```powershell
build\dev\Debug\xvram-probe.exe
build\dev\Debug\xvram-probe.exe --json report.xvram-report.json
build\dev\Debug\xvram-probe.exe --overlap --json overlap.xvram-report.json
build\vs\Debug\xvram-probe.exe
build\vs\Debug\xvram-probe.exe --json report.xvram-report.json
```

On Linux:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
./build/dev/xvram-probe --json report.xvram-report.json
```

Run the Phase 1 proof on a VMM-capable CUDA device:

```powershell
build\vs\Release\xvram-vmm-poc.exe --json vmm-poc.json
build\vs\Release\xvram-vmm-poc.exe --logical-size 12GiB --mode both --json acceptance.json
```

```bash
./build/dev/xvram-vmm-poc --json vmm-poc.json
```

Run the Phase 2 cache suite on a VMM-capable CUDA device:

```powershell
build\vs\Release\xvram-cache-bench.exe --scenario suite --policy both --json cache-suite.json
build\vs\Release\xvram-cache-bench.exe --scenario suite --policy both `
  --trace cache-suite.jsonl --json cache-suite.json
```

```bash
./build/dev/xvram-cache-bench --scenario suite --policy both --json cache-suite.json
```

Install and consume the Phase 3 C ABI:

```powershell
cmake --install build\vs --config Release --prefix build\install
```

An installed CMake consumer uses `find_package(xVRAM CONFIG REQUIRED)` and links
`xVRAM::xvram` or `xVRAM::torch_allocator`; plain C consumers can include
`<xvram/xvram.h>`, opt into compression with `<xvram/xvram_v2.h>`, or include
`<xvram/torch_allocator.h>`.

Run the Phase 3 GEMM gate on a CUDA/cuBLAS-capable device:

```powershell
build\vs\Release\xvram-gemm-bench.exe --suite --json gemm-suite.json
```

```bash
./build/dev/xvram-gemm-bench --suite --json gemm-suite.json
```

Install and smoke-test the Phase 4a Python adapter:

```powershell
python -m pip install .\python
$env:XVRAM_TORCH_ALLOCATOR_LIBRARY = `
  (Resolve-Path .\build\vs\Release\xvram_torch_allocator.dll).Path
python .\tests\python\test_torch_allocator_integration.py -v
```

The reproducible Windows gate is `scripts/run-phase4-pytorch-acceptance.ps1`.

Build and locate the private Phase 4b runtime with the same Python interpreter that
provides compatible PyTorch Stable-ABI headers and libraries:

```powershell
python -m pip install .\python
cmake -S . -B build\vs -A x64 `
  -DXVRAM_BUILD_TESTS=ON -DXVRAM_BUILD_TORCH_RUNTIME=ON
cmake --build build\vs --config Release `
  --target xvram_torch_runtime xvram-torch-runtime-control-tests
$env:XVRAM_TORCH_RUNTIME_LIBRARY = `
  (Resolve-Path .\build\vs\Release\xvram_torch_runtime.dll).Path
python -m xvram.torch_bench --help
```

The reproducible Windows hardware matrix is encoded by
`scripts/run-phase4b-rtx3070-acceptance.ps1`. Running the script is intentionally not
part of a normal build: it constructs and verifies multi-gigabyte models and requires
the specified RTX 3070/PyTorch/CUDA environment.

Run the Phase 5 core benchmark on a VMM-capable CUDA device:

```powershell
build\vs\Release\xvram-compression-bench.exe `
  --logical-size 24GiB --scenario compressible-read `
  --compression-policy adaptive --path all --policy clock `
  --trace compression.jsonl --json compression.json
```

Explicit sizes use ordinary byte suffixes in the CLI; the acceptance script derives the
exact `3x` byte count from `xvram-probe` and passes that value. The reproducible Windows
core matrix is `scripts/run-phase5-compression-acceptance.ps1`. It checks schemas,
transfer/event ordering, accounting, reference equality, cleanup, and worker exit.
After that core gate, `scripts/run-phase5-sdk-v2-hardware-smoke.ps1` exercises the
public v2 function table, and `scripts/run-phase5-pytorch-acceptance.ps1` runs the
compression-off regression matrix followed by the PyTorch v2 cases. See the
[reproduction commands](docs/adaptive-compression.md#reproducing-the-gates).
These are manual hardware gates; their scripts alone are not completion evidence.

The first configure may require network access for the hash-pinned NVIDIA headers.
For an offline build, install CUDA 13.x and NVML development headers (or set
`XVRAM_CUDA_INCLUDE_DIR` and `XVRAM_NVML_INCLUDE_DIR`) and configure with
`-DXVRAM_FETCH_CUDA_HEADERS=OFF`. The CUDA Runtime headers must include their
matching `crt/` headers; if stored separately, set `XVRAM_CUDA_CRT_INCLUDE_DIR`
to the include root containing `crt/host_config.h`. The automatic CUDA 13.3 path
fetches both `cuda_cudart 13.3.29` and `cuda_crt 13.3.33` with pinned hashes.

Use `xvram-probe --help`, `xvram-vmm-poc --help`, `xvram-cache-bench --help`,
`xvram-gemm-bench --help`, `xvram-compression-bench --help`, and
`xvram-torch-bench --help` for the complete CLI contracts. After installing the Python
package, `xvram-compat-audit --help` describes the separate diagnostic workflow.

`--overlap` is explicit and bounded. It calibrates a short compute-only workload, balances
independent H2D and D2H batches to a similar duration, then reports their concurrent
makespan, speedup, and overlap efficiency. `--require-overlap` exits with code 22 unless
the compute result, copied bytes, and cleanup all pass.

Default reports redact UUID, LUID, and PCI bus identifiers, but still describe exact
hardware, software, memory, and live budget information. They are sanitized rather than
anonymous; review [`PRIVACY.md`](PRIVACY.md) before sharing one.

## Roadmap

1. Hardware and driver capability probe.
2. Explicit CUDA VMM proof of concept with stable logical addresses.
3. Event-safe residency cache with pinned staging pools and WDDM budget tracking.
4. Stable C ABI and library-aware tiled operations, starting with GEMM.
5. PyTorch resident allocator plus lease-scoped static inference; both Phase 4
   boundaries and the Phase 4b oversubscription gate are complete.
6. Adaptive lossless compressed host backing with SDK/PyTorch v2 opt-in; implementation,
   69 local tests, 12 core hardware runs, SDK v2 smoke, and 11 PyTorch hardware runs passed.
   The [full Phase 5 CI matrix](https://github.com/TheLitis/xVRAM/actions/runs/33928261245)
   also passed; Phase 5 is complete within its stated scope.
7. Explicit synchronous CUDA/cuBLAS integration (Phase 6a), followed by an evidence-led
   unchanged-application compatibility audit (Phase 6b.0).
8. Conservative CUDA interception of a proven profile, followed by broader kernel
   coverage and PTX access instrumentation. The audit itself is not this execution path.

See [the architecture](docs/architecture.md), [memory model](docs/memory-model.md),
[correctness rules](docs/correctness.md), [VMM proof](docs/vmm-poc.md),
[residency cache](docs/residency-cache.md), and [roadmap](docs/roadmap.md) for the design
contract. The [PyTorch MemPool](docs/pytorch-mempool.md) and
[lease-scoped inference](docs/pytorch-inference.md) guides describe the two distinct
Phase 4 integration boundaries. The [adaptive-compression guide](docs/adaptive-compression.md)
describes the Phase 5 backing and transport boundary.

## Performance expectations

xVRAM cannot make PCIe-attached RAM as fast as GDDR. A workload that already fits in
VRAM should use a near-zero-overhead bypass path and will remain fastest with ordinary
CUDA allocation. The opportunity is to:

- make an otherwise out-of-memory workload execute;
- transfer fewer bytes than opaque fallback or naive offload;
- keep reusable data resident;
- overlap remaining transfers with useful GPU work;
- use graph or library knowledge to avoid cache thrashing.

## Project status and license

The ABI v1 function table and report schemas are pre-release and may change before
`0.1.0`.
A project license has not yet been selected; until one is added, no rights are granted
beyond those provided by applicable law. Keep the GitHub repository private until that
choice is made.
