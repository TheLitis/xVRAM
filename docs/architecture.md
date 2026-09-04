# Architecture

## Goal

xVRAM provides a logical CUDA heap larger than physical VRAM while preserving GPU-only
execution for the main numerical work. The complete canonical data set lives in system
RAM. A bounded physical VRAM working set is treated as a software-managed write-back
cache.

The architecture is deliberately split into a universal correctness layer and optional
knowledge layers. Correctness must never depend on a prediction being right. Framework,
library, and compiler integrations improve scheduling and transfer volume.

## Layers

```text
Framework adapters / public C ABI / known operations
                          |
                 working-set contract
                          |
                  logical GPU heap
                          |
       residency manager + synchronization
             /             |             \
     cache policy     transfer engine     telemetry
             \             |             /
          CUDA VMM / Driver API / WDDM budgets
                  /                 \
               VRAM              host RAM
```

### Integration layer

The best integration declares allocation lifetime, access mode, next use, and the exact
range required by an operation. Generic interception can only provide conservative
allocation-level residency until access analysis or instrumentation is available.

Phase 3 exposes that contract through the versioned C function table returned by
`xvram_get_api`. The shared library keeps its C++ implementation private and represents
sessions, allocations, plans, and operations with opaque handles. Every public structure
has a size-tagged v1 prefix so later ABIs can reject an incompatible caller without
reading beyond the caller-owned object.

The generic transaction boundary resolves declared ranges and invokes trusted client
code on a session-owned worker thread. The session CUDA context is current and one
runtime stream is supplied for enqueue-only use. Resolved device addresses and the
workspace are borrowed only until callback return; retaining them would bypass residency
and event-generation ownership.

### Logical heap

The heap owns stable CUDA virtual-address reservations. Logical allocation identity is
separate from physical memory handles and from host backing storage. A remap is legal
only at a synchronization boundary where no in-flight operation can access the range.

The Phase 2 implementation assigns monotonic allocation IDs. Every allocation has its
own padded VA reservation; cache metadata addresses chunks as
`{allocation_id, chunk_index}`. A physical frame owns one reusable VMM handle and at
most one mapping, so allocation lifetime, logical address lifetime, and frame lifetime
remain independent. The frozen Phase 2 benchmark keeps its original flat pageable
backing as a regression boundary. The production runtime replaces that flat image in
Phase 5 with one authoritative `HostBackingStore` entry per chunk.

Phase 3 retains this heap behind the public ABI. Allocation priorities (`normal`, `hot`,
and `streaming`) are eviction-cost hints, not permanent residency promises. Host
read/write functions operate on canonical backing and are serialized through the session
worker so they cannot race conflicting GPU ownership.

### Authoritative host backing

The Phase 5 production backing store decouples logical size from committed raw host
pages. A chunk has exactly one authoritative representation and content generation:

```text
invalid | implicit_zero | raw | lz4_blocks
```

`xvram_lz4_blocks_v1` divides the default 64 MiB residency chunk into independent
64 KiB blocks. Each block is zero-elided, stored raw, or encoded as an ordinary LZ4
block. The container records `valid_bytes`, generation, sizes, tags, and a 128-bit
content token. A compressed block is accepted only when smaller than its raw form, so
the container never claims capacity by expanding incompressible data and remains fully
decodable by the pinned CPU LZ4 implementation.

Conversions build an immutable candidate generation under conversion-scratch budget,
decode and verify its token, and then atomically replace the previous authority. The
previous generation remains valid until commit. A complete host write can construct a
new generation without reading the old one. A partial host write decodes only the
affected blocks and preserves the untouched blocks. `discard_dead` may invalidate a
proven-dead activation after its final event; ordinary release still preserves dirty
data through write-back.

Before a write-capable GPU lease first dirties a chunk, the host ledger reserves a raw
spill credit large enough for that chunk. The kernel is not launched if that reservation
cannot be made. This admission rule means an incompressible result can always fall back
to raw D2H without replacing the last valid host generation. There is no disk spill:
host-budget exhaustion is an explicit OOM boundary, never partial authority.

### Residency manager

The manager is the correctness authority. It resolves a requested working set into
physical mappings, schedules dirty write-back, waits on the necessary events, and only
then authorizes launch. It also reacts to WDDM budget changes by shrinking its target
cache size.

Its internal transaction boundary accepts normalized read, read-write, and write-only
ranges. A full-chunk write-only access skips H2D, while a partial write-only access first
loads the chunk to preserve untouched bytes. Only one compute transaction executes at a
time; prefetch and write-back may use their independent transfer resources around it.

The implemented scheduling order is deliberate:

1. retire completed event generations;
2. react to a live-budget shrink;
3. schedule required dirty write-back;
4. service demand H2D;
5. launch the declared compute transaction;
6. fill remaining H2D capacity with speculative prefetch.

One worker thread owns the CUDA call sequence for each public session. Isolated mode
creates and destroys a private context there. Attach-current mode captures the caller's
current context, pushes it for worker activity, pops it at the boundary, and never
destroys the caller-owned context. Multiple operations may be queued, but at most one
compute transaction executes at once.

### Transfer engine

The transfer engine uses a bounded pool of pinned staging memory. The complete backing
store remains pageable unless a backend proves a better choice. H2D, D2H, and compute
streams are kept distinct when the device exposes useful copy engines.

Phase 2 divides the default four pinned buffers into two H2D and two D2H staging slots.
Each slot has generation ownership: it cannot be reused until the matching event is
successfully retired and, for D2H, the bytes have been copied into pageable backing.

Phase 5 adds generation-safe codec slots and distinct encode/decode streams around that
same production transfer manager. The three paths are:

```text
raw:
  host raw -> pinned H2D -> stable mapped VA

cpu_lz4_gpu_decode:
  host lz4_blocks -> compressed H2D -> nvCOMP CUDA decode -> stable mapped VA

nvcomp_gpu_codec write-back:
  compute event -> nvCOMP CUDA encode -> compressed D2H -> verify -> atomic host commit
```

A raw fallback is allowed only while a valid authority and the necessary spill
reservation remain intact. A codec slot names the operation, chunk, source generation,
and its own monotonically increasing generation. It is reusable only after
`CUDA_SUCCESS`, host verification, and commit. `CUDA_ERROR_NOT_READY` is the only other
accepted polling result. Once submission state is unknown, the runtime quarantines the
slot, mapping, handle, reservation, and worker instead of guessing whether reuse is safe.

GPU decode writes into the already mapped stable logical VA and is followed by content
verification before the chunk becomes resident. GPU encode precedes PCIe D2H. On an
Ampere-class device nvCOMP uses its CUDA backend; no hardware Decompression Engine is
assumed.

CPU encoding compares adjacent blocks in a budgeted, verified source snapshot. An
identical block reuses its predecessor's retired encoded result, including across CPU
queue batches. Each container block still owns its payload and participates in full
candidate verification; this reduces repeated codec work without introducing a shared
storage dictionary or changing the format.

### Policy

Policy may combine explicit next-use distance, hotness, reuse count, cost to reload,
dirty state, and allocation class. Policy selects victims but cannot violate pinning,
in-flight, or working-set constraints.

The implemented default is cost-aware CLOCK. Demand access sets a reference bit,
speculative insertion is cold, clean candidates are preferred to dirty candidates, and
sequential one-touch data is biased toward early eviction. Deterministic LRU is the
pluggable baseline, with `ChunkKey` as its final tie-breaker. The residency manager
revalidates every proposed victim against pinning, in-flight ownership, and the current
working set.

Compression selection is separate from eviction policy. Its per-content-generation
EWMA observes ratio, CPU/GPU encode and decode cost, PCIe bandwidth, staging and CPU
queue latency, reuse, dirty rate, GPU occupancy, and SM opportunity cost. Two samples
are required before a path is considered calibrated. In `adaptive` mode a compressed
path must beat raw by at least `max(10%, 50 us)` end to end; three unfavorable probes
mark only the current content generation `never_compress`. `capacity` mode may prefer a
smaller lossless representation despite a time penalty. A content change creates a new
generation and therefore a new decision history.

## Live target

The cache target is recomputed with saturating arithmetic from reclaimable CUDA memory,
the configured cap, and, on Windows, reclaimable WDDM local-memory budget:

```text
cuda_reclaimable = cuda_free + managed_frame_bytes
wddm_reclaimable = (Budget - CurrentUsage) + managed_frame_bytes

target = floor_to_chunk(
  min(configured_cap, cuda_reclaimable, wddm_reclaimable) - device_headroom
)
```

Linux omits the WDDM term. Shrink stops new launches, drains the active transaction,
writes dirty chunks back, removes excess mappings, and releases excess frames. Grow
requires ten consecutive safe samples with at least two chunks of surplus and adds at
most two chunks per control cycle. One allocation OOM may trigger one refresh, shrink,
and retry.

Runtime-owned CUDA-library workspace is reserved outside the cache-frame pool but is
charged against the same configured cap, live CUDA/WDDM observations, and device
headroom. A budget shrink blocks new tiles, drains the active tile, writes dirty chunks
back, and releases excess mappings/frames. If the next tile no longer fits, execution
returns budget pressure at that safe boundary; the caller can create a smaller plan.

Phase 5 adds nvCOMP workspace, device codec slots, status/token buffers, and resident
compute scratch to the device-side charge. The host ledger independently charges
authoritative raw/compressed images, conversion scratch, pinned staging, and spill
credits. During shrink the runtime stops new work, retires in-flight generations, then
releases codec buffers and frames. An explicit logical size is never clamped to a fixed
VRAM multiple: admission is determined by CUDA VA, live host storage, the declared
working set, and live device/WDDM budgets. Automatic sizing deliberately remains the
raw-safe `1.5x VRAM` default.

## Known-operation layer

The Phase 3 GEMM path converts a matrix operation into a deterministic sequence of
explicit residency transactions. The planner enumerates M/N/K tiles, derives exact
strided A/B/C byte ranges for layout and transpose, and accepts only candidates whose
unique chunks plus workspace fit the minimum live target. K panels for one C tile stay
contiguous: the first panel applies the user beta and later panels accumulate with beta
equal to one. Event-safe write-back/reload preserves correctness if C does not remain resident.

cuBLASLt is the preferred executor. Its heuristic selection is cached by complete tile
signature, including dimensions, data types, layouts, transposes, compute mode, and
workspace limit. If no compatible Lt algorithm exists, the executor falls back to the
matching cuBLAS GEMM entry point. Strict FP16/BF16-output operations deliberately use
the pedantic core cuBLAS path with reduced-precision reduction disabled and an unsplit K
dimension. This is a known-operation wrapper; arbitrary cuBLAS calls are not intercepted.

## Backend strategy

The portable core uses the CUDA Driver API. CUDA Driver, cuBLAS, and cuBLASLt symbols
are resolved dynamically, so the SDK does not link an application directly to the CUDA
Runtime API. Each cuBLAS handle is created and destroyed on its session worker thread
while that session's context is current. The Windows backend adds DXGI/WDDM budget
observation and LUID-based adapter matching. Host VMM is an optional capability, not a
requirement for the resident-only baseline: pageable backing plus a bounded pinned
staging pool remains the compatibility path.

All optional behavior is capability-gated at runtime. A recent header or driver version
does not imply that a particular GeForce implements every location or memory-pool mode.

The CPU block codec is hash-pinned LZ4 1.10.0. The optional GPU path uses the official,
hash-pinned nvCOMP 5.3.0.16 CUDA 13 redistributable. Its shared libraries are staged and
installed beside xVRAM, and the codec dispatch resolves that app-local copy dynamically.
Configure verifies the GPU shared library against the platform SHA-256 pin. The loader
hashes its resolved file before and after loading and requires the exact 5.3.0 runtime
semantic version; the file pin identifies package build 5.3.0.16. Reports use that
verified file hash rather than merely repeating an expected build constant. ABI v1 and
compression-off PyTorch sessions do not initialize nvCOMP. This preserves no-driver and
no-codec diagnostics and permits relocation without a machine-wide nvCOMP installation.

## Public boundaries

- A size-tagged, versioned C ABI for framework adapters and known operations.
- A C++ implementation API hidden behind the ABI.
- Versioned JSON telemetry and capability reports.
- Explicit working-set transactions for known libraries.
- Conservative allocation-level transactions for generic interception.

`xvram-probe` validates backend capabilities and transport assumptions.
`xvram-vmm-poc` is the deliberately narrow stable-address FIFO proof.
`xvram-cache-bench` is the Phase 2 boundary: its isolated worker drives the internal
multi-allocation residency engine through deterministic workloads and returns telemetry
to a controller-owned report and trace sink. The `xvram` shared library and
`<xvram/xvram.h>` are the Phase 3 SDK boundary. `xvram-gemm-bench` keeps hardware
acceptance outside the caller process through the `XVG1` controller/worker protocol;
that process isolation is a benchmark safety boundary, not a property of an SDK callback.

Phase 5 adds `<xvram/xvram_v2.h>` without changing `<xvram/xvram.h>` or introducing a
second SDK export. API v2 embeds the complete v1 table and adds only compression-aware
session creation and telemetry. The private PyTorch control table follows the same
prefix rule; `_wrap_resolved_v1` remains unchanged. `xvram-compression-bench` is the
isolated core acceptance boundary and speaks `XVZ1`, while compression-enabled PyTorch
uses the v2 report/trace/protocol family and compression-off execution stays on the
frozen Phase 4b v1 contracts.

The core JSONL trace records actual transfer submissions and retirements with allocation,
chunk, operation, source-generation, and slot-generation identities. Its validator checks
H2D/decode and encode/D2H ordering, rejects reused generations and missing retirements
in completed runs, and reconciles codec submissions with report counters. PCIe totals
include both payload and metadata, which are reported separately.
