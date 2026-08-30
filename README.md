# xVRAM

**xVRAM** is an experimental, Windows-first runtime for software-managed tiered CUDA
memory. System RAM is the large capacity and backing tier; physical VRAM is the fast,
write-back residency tier where kernels execute.

The project is aimed at workloads whose total data is larger than VRAM but whose
per-operation working set can be made resident. The runtime combines stable CUDA
virtual addresses, asynchronous transfers, explicit working-set declarations,
WDDM-aware budgeting, cache policy, and library-aware tiling.

> [!WARNING]
> xVRAM now exposes an experimental, versioned C ABI and a tiled GEMM integration,
> but the SDK remains pre-release and does not transparently extend an arbitrary
> application's VRAM. It must not be used for production workloads.

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
compatibility fallback. FP16, BF16, FP32/TF32, and FP64, row/column-major layouts, and
N/T operands are represented by ABI v1. Batching, complex values, fused epilogues,
framework adapters, and transparent CUDA interception remain out of scope.

`xvram-gemm-bench` is the isolated controller/worker hardware gate for this path. It
validates tiled numerical output, reports algorithms, workspace and residency telemetry,
and writes the strict `xvram.gemm_bench` v1 report without serializing raw CUDA virtual
addresses.

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
`xVRAM::xvram`; plain C consumers can include `<xvram/xvram.h>` and dynamically resolve
`xvram_get_api`.

Run the Phase 3 GEMM gate on a CUDA/cuBLAS-capable device:

```powershell
build\vs\Release\xvram-gemm-bench.exe --suite --json gemm-suite.json
```

```bash
./build/dev/xvram-gemm-bench --suite --json gemm-suite.json
```

The first configure may require network access for the hash-pinned NVIDIA headers.
For an offline build, install CUDA 13.x and NVML development headers (or set
`XVRAM_CUDA_INCLUDE_DIR` and `XVRAM_NVML_INCLUDE_DIR`) and configure with
`-DXVRAM_FETCH_CUDA_HEADERS=OFF`.

Use `xvram-probe --help`, `xvram-vmm-poc --help`, `xvram-cache-bench --help`, and
`xvram-gemm-bench --help` for the complete CLI contracts.

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
5. PyTorch allocator and graph-scheduling adapter.
6. Adaptive transport compression based on measured cost.
7. Conservative CUDA interception, followed by PTX access instrumentation.

See [the architecture](docs/architecture.md), [memory model](docs/memory-model.md),
[correctness rules](docs/correctness.md), [VMM proof](docs/vmm-poc.md),
[residency cache](docs/residency-cache.md), and [roadmap](docs/roadmap.md) for the design
contract.

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
