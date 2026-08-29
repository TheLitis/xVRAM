# xVRAM

**xVRAM** is an experimental, Windows-first runtime for software-managed tiered CUDA
memory. System RAM is the large capacity and backing tier; physical VRAM is the fast,
write-back residency tier where kernels execute.

The project is aimed at workloads whose total data is larger than VRAM but whose
per-kernel working set can be made resident. The runtime will combine stable CUDA
virtual addresses, asynchronous transfers, explicit working-set declarations,
WDDM-aware budgeting, cache policy, and framework hints.

> [!WARNING]
> xVRAM is at the hardware-probing stage. It does not yet extend an application's VRAM
> and must not be used for production workloads.

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

## Build

Requirements:

- 64-bit Windows 10/11 or a 64-bit Linux development host;
- CMake 3.25 or newer;
- a C++20 compiler (MSVC 2022/2026, Clang, or GCC).
- Python 3 plus `tests/requirements.txt` when running the JSON Schema contract test.

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

The first configure may require network access for the hash-pinned NVIDIA headers.
For an offline build, install CUDA 13.x and NVML development headers (or set
`XVRAM_CUDA_INCLUDE_DIR` and `XVRAM_NVML_INCLUDE_DIR`) and configure with
`-DXVRAM_FETCH_CUDA_HEADERS=OFF`.

Use `xvram-probe --help` for the complete CLI contract.

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
4. Library-aware tiled operations, starting with GEMM.
5. PyTorch allocator and graph-scheduling adapter.
6. Adaptive transport compression based on measured cost.
7. Conservative CUDA interception, followed by PTX access instrumentation.

See [the architecture](docs/architecture.md), [memory model](docs/memory-model.md),
[correctness rules](docs/correctness.md), and [roadmap](docs/roadmap.md) for the design
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

The public API and report schema are pre-release and may change before `0.1.0`.
A project license has not yet been selected; until one is added, no rights are granted
beyond those provided by applicable law. Keep the GitHub repository private until that
choice is made.
