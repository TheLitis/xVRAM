# Explicit synchronous CUDA/cuBLAS compatibility

Phase 6a provides a deliberately limited source-integration profile. A host-only C++
application is rebuilt with `<xvram/cuda_compat.hpp>` and `xVRAM::cuda_compat`. xVRAM
derives residency ranges for supported operations; the application does not declare
transactions. This is not a replacement for CUDA Runtime and does not run unchanged
executables, arbitrary kernels, or third-party CUDA libraries transparently.

## Build and integrate

The profile is optional and defaults to OFF. It requires official NVIDIA Runtime and
cuBLAS headers but does not link the consumer to cudart, cuBLAS, or Driver libraries.
The existing dynamic loader and its system/explicit/app-local resolution remain unchanged.

```sh
cmake -S . -B build -DXVRAM_BUILD_CUDA_COMPAT=ON -DXVRAM_FETCH_CUBLAS_REDIST=ON
cmake --build build --config Release --parallel
cmake --install build --config Release --prefix install
```

The cuBLAS redistributable is the existing hash-pinned `13.5.1.27` core/Lt pair. The
configure step records the staged library hashes in `compat-cublas-provenance.json`.
An installed Toolkit may supply the headers instead. Applications supply those official
header include directories as their own build dependencies; xVRAM does not re-export
build-machine Toolkit paths in its installed CMake targets.

Without an installed Toolkit, header fetching supplies the matching Runtime, CRT,
and CCCL packages. The latter provides `nv/target`, included transitively by cuBLAS
even for FP32-only host code. CCCL `13.3.3.3.1` is SHA-256-pinned to NVIDIA's
[CUDA 13.3 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json).
Custom header layouts can set `XVRAM_CUDA_CRT_INCLUDE_DIR` and
`XVRAM_CUDA_CCCL_INCLUDE_DIR`; configure verifies that the complete facade compiles.

```cmake
project(my_inference LANGUAGES CXX)
find_package(xVRAM CONFIG REQUIRED)
add_executable(my_inference main.cpp)
target_link_libraries(my_inference PRIVATE xVRAM::cuda_compat)
target_include_directories(my_inference SYSTEM PRIVATE "${NVIDIA_INCLUDE_DIR}")
```

The C control header negotiates one size-tagged API table. Initialize explicitly before
facade calls, check every result, and call shutdown while the application is still alive:

```cpp
#define XVRAM_CUDA_COMPAT_REMAP_NAMES 1
#include <xvram/cuda_compat.hpp>

xvram_cuda_compat_api_v1 api{};
auto status = xvram_cuda_compat_get_api(
    XVRAM_CUDA_COMPAT_ABI_VERSION_1, sizeof(api), &api);
if (status != XVRAM_STATUS_SUCCESS) return 1;
xvram_cuda_compat_config_v1 config = XVRAM_CUDA_COMPAT_CONFIG_V1_INIT;
if (api.initialize(&config) != XVRAM_STATUS_SUCCESS) return 2;
// Checked cudaMalloc, explicit cudaMemcpy and cublasSgemm calls go here.
const auto closed = api.shutdown();
return closed == XVRAM_STATUS_SUCCESS ? 0 : 3;
```

Alternatively omit the remapping macro and call names such as
`xvram::cuda_compat::cudaMalloc` explicitly. There is exactly one valid initialization
attempt per process; shutdown cannot be followed by reinitialization in v1. Invalid
configuration is rejected before consuming that attempt. `get_error` gives the detailed
last adapter failure and `get_telemetry` gives a coherent snapshot during blocking work
and after shutdown. CUDA last-error facade state is thread-local; peek preserves it and
get clears it. Successful calls do not erase a previous CUDA error.

## Supported boundary

| Family | Supported | Rejected |
| --- | --- | --- |
| Memory | `cudaMalloc`, exact-base `cudaFree`, explicit H2D/D2H `cudaMemcpy` | Foreign/stale device pointers, D2D, default-kind inference |
| Device | Count/get, set to the initialized device, synchronize, last-error helpers | Switching GPU, async/stream interoperability, capture/graphs |
| cuBLAS handle | Create/destroy/version, host pointer mode, default stream | Foreign/stale handles, device scalars, non-default streams |
| Math | Default or pedantic mode, both strict FP32 | TF32/tensor-op modes, hidden native fallback |
| GEMM | Positive M/N/K, column-major FP32 SGEMM, N/T, padding/interior pointers | Zero dimensions, conjugate transpose, C/input overlap, nonfinite alpha/beta |

For supported calls, invalid arguments fail before GPU submission. Unsupported source
calls retain their NVIDIA declarations and fail compilation/linking in the tested
host-only profile because no native CUDA libraries are linked. Device compilation is
explicitly rejected by the facade. This integration contract is not a sandbox: a library
which independently loads CUDA or submits work outside the facade is out of scope.

GEMM dimensions and leading dimensions follow the official
[cuBLAS GEMM contract](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm).
Alpha/beta are captured before worker submission. Only the first K panel applies the
caller beta; subsequent panels accumulate with beta one. Padding and bytes outside the
declared C result are preserved. Both supported math modes prohibit implicit TF32.

## Memory, scheduling, and failures

A serial worker owns an isolated context and its cuBLAS/Lt resources. Physical compute
uses the production residency runtime and shared tiled GEMM executor. There is no
second transfer manager. All changing facade calls complete before returning.
As with native C APIs, host arguments must designate valid, live host objects for the
call's duration. The registry rejects known managed ranges and tombstones; it is not
an arbitrary host-pointer accessibility probe.

`cudaMalloc` reserves a stable logical VA and raw pageable backing. H2D copies finish
into authoritative host backing; GPU upload occurs on demand at GEMM. D2H obtains the
latest result through the runtime read/write-back path. Compression is disabled here;
the separate SDK/PyTorch v2 compressed profiles are unchanged.

Freed allocations release backing and mappings normally but retain their exact VA
reservation until shutdown. Tombstones prevent stale pointers from resolving to later
allocations. This consumes VA, not VRAM or backing RAM. Reservation exhaustion fails
allocation; it never enables address reuse. The cleanup ledger reserves capacity before
resource release. Default SDK allocation lifetime remains unchanged.

Every map is followed by SetAccess. Full mappings are unmapped only after completed
event boundaries. Dirty reuse requires completed write-back. Live budget changes retain
the runtime's shrink/grow and WDDM rules. Defaults are device 0, 64 MiB aligned chunks,
automatic cache target, 512 MiB device headroom, 4 MiB workspace, four staging buffers,
CLOCK, and prefetch distance two. Host admission reserves the greater of 4 GiB and 25%
physical RAM, plus charged staging and implementation scratch.

A region exceeding 250 ms finishes safely but prevents the next launch. The separate
5,000 ms event-stall timeout does not raise that 250 ms limit. An execution error after
submission may leave partial C, and poisons the session. Unknown completion retains
the runtime's quarantine boundary. The library does not kill its caller; applications
must not continue using a poisoned session. Shutdown attempts all safe cleanup stages
on the owning worker, including when allocating a cleanup queue entry fails.

No CUPTI callback performs residency work. No driver, TDR, registry, or NVIDIA settings
are changed. Profiling hooks and asynchronous/native-stream integration remain outside
this phase; see NVIDIA's [callback restrictions](https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api).

## Benchmark and contracts

`xvram-compat-bench --help` lists geometry, N/T, padding, offset, alpha/beta, policy,
cache/chunk/headroom/workspace, trace, JSON, and timeout controls. Scenarios are `suite`,
`gemm`, and `rejection`. Logical auto uses the existing raw-safe maximum of 1.5x VRAM;
explicit sizes have no multiplier clamp. Exact geometry cannot be combined with a
logical-size override. Logical sizing uses M=4096, N=16 and rounds down to a complete
K plane, reporting the actual A+B+C operand bytes separately from storage padding.

```sh
xvram-compat-bench --scenario suite --m 37 --n 19 --k 67 \
  --padding 7 --offset-elements 17 --alpha 1.25 --beta 0.5 \
  --json small.json --trace small.jsonl
```

The controller owns text/JSON/trace files. Its isolated worker uses bounded `XVI1`
frames (1 MiB payload; 64 KiB trace batches). Heartbeats arrive once per second but do
not reset the 15-second no-progress watchdog. Actual copies, residency lifecycle work,
and retired tiles provide progress. The default total deadline is 900 seconds. Windows
Job Objects and Linux process groups contain the worker; timeout terminates and reaps
it. Failure reports leave unverified proof/cleanup flags null rather than claim success.

New strict contracts are `xvram.cuda_compat` v1 and `xvram.cuda_compat_trace` v1. Reports
and traces contain no raw addresses or native handles. Device identifiers are hidden
by default. API H2D/D2H byte counts are logical facade copies and are separate from
physical cache transfers. Tile trace timestamps are host observations of retirement,
not fabricated GPU timestamps. At quiescence, calls attempted equal completed plus
rejected; submitted counts worker-dispatched calls, including argument validation, and
must not be confused with GPU launches. Current retained-reservation counters go to
zero at shutdown; freed counters are cumulative.
Runtime `event_boundaries` counts safe unmaps, so it must equal `unmaps` on completion.
Compute retirement is reconciled separately: `transactions_completed == tiles_retired`.
Cache-hit GEMMs can retire many tiles while keeping the same few mappings resident.

Exit codes retain the project family: 0 proof complete; 23 prerequisite/configuration;
24 mismatch; 25 OOM/budget pressure; 26 timeout; 27 CUDA/platform/protocol/cleanup;
64 usage; 70 internal; 74 output I/O. All Phase 1–5 ABI/schema/export contracts stay frozen.

## Validation and acceptance

No-driver tests cover registry ranges/ABA, scalar capture, worker ownership/concurrency,
injected failures, stale handles, cleanup, export negotiation, supported/negative host
consumers, protocol crash/hang/heartbeat-only hang, framing, report semantics, and reap.
CI runs compatibility OFF then ON on Windows/Linux Debug/Release, ON under Linux
Clang ASan/UBSan, installed C/facade consumers, and relocated app-local cuBLAS dispatch.
The previous PyTorch Stable-ABI 2.11→2.13 regression remains enabled.

The reproducible hardware gate is `scripts/run-phase6a-rtx3070-acceptance.ps1`, requiring
an explicit core/Lt library pair. Its manifest records source/binary/library hashes.
It covers padded NN/NT/TN/TT plus ordinary cuBLAS and full CPU FP64 reference, constrained
split K, and 1.1x/1.5x/2.0x CLOCK/LRU two-pass oversubscription. Large operands are
initialized in bounded chunks; every output is checked analytically without a second
full raw input copy. The criterion is `abs(error) <= 1e-4 + 2e-5 * abs(reference)`.
Padding is checked bitwise. Digests are recorded, not required to match across valid
FP32 reduction orders. Oversized cases require real eviction and frame reuse, balanced
maps/SetAccess/unmaps, zero unsafe activity, complete cleanup, and no residual worker.
There is no speedup threshold.

### Recorded local results

The full RTX 3070 gate completed on 2026-09-05 at signed source revision
`d6e5ded`, against 8,589,410,304 bytes of physical VRAM. All eight reports and their
traces passed the strict contracts, numerical reference, accounting, cleanup, and
worker-reaping checks. The small suite contains five padded cases, including NN/NT/TN/TT
with full CPU FP64 and ordinary cuBLAS verification. Constrained split K retired 128
tiles across two passes and preserved every checked padding byte.

The following counts were identical for CLOCK and LRU at each size:

| Ratio | Actual operand bytes | Retired tiles | Maps = SetAccess = unmaps | Evictions / handle reuse |
| --- | ---: | ---: | ---: | ---: |
| 1.1x | 9,448,338,752 | 1,122 | 284 | 187 |
| 1.5x | 12,884,112,128 | 1,530 | 386 | 289 |
| 2.0x | 17,178,816,512 | 2,040 | 512 | 415 |

Policy output digests also agreed at each ratio. Maximum absolute FP32 errors were
0.0625, 0.0625, and 0.125 respectively; every output satisfied the stated combined
absolute/relative criterion. Mismatch and unsafe-activity counts were zero, every
proof/cleanup flag was true, diagnostics were empty, and no worker remained. These
ratios are acceptance points, not fixed product capacity limits.

The selected app-local redistributable pair was explicitly passed to both adapter and
ordinary-cuBLAS baseline: version `13.5.1.27` (reported cuBLAS API version `130501`).
SHA-256: core `f1d500d0cd892f5b8c6b6cdbffd82d0c55d5f5427215668e7ceb55aeeccc1b63`,
Lt `b592cd016d7673e9cb97716a22b27c4010ee635377a3ba28f37070a9bdb76a68`.
Full source/binary/library provenance and per-run timings are recorded in
`artifacts/phase6a-rtx3070-release-d6e5ded-20260905/acceptance-manifest.json`.

Local no-driver validation passed Release 79/79, Debug 76/76, and installed consumers
6/6, plus relocated app-local library hashes/dispatch. The Release build includes the
PyTorch Stable-ABI bridge regression. Focused counter regressions validate many cache-hit
transactions against fewer unmap boundaries. Standalone Runtime/CRT/CCCL header builds
and platform-specific missing-executable/reap behavior were also verified during CI
integration, without changing the GPU execution path.

The complete remote Windows/Linux Debug/Release, compatibility OFF/ON, Clang ASan/UBSan,
installed-consumer/relocation, and PyTorch Stable-ABI matrix passed at signed revision
`55def7b01d5748bfd77adb9a18f674f6e583ec4f` in
[GitHub Actions run 33960302667](https://github.com/TheLitis/xVRAM/actions/runs/33960302667).
This completes Phase 6a within its explicitly synchronous, rebuilt-application scope.
