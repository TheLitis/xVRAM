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
Framework adapters / CUDA library wrappers / interception
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

### Logical heap

The heap owns stable CUDA virtual-address reservations. Logical allocation identity is
separate from physical memory handles and from host backing storage. A remap is legal
only at a synchronization boundary where no in-flight operation can access the range.

The Phase 2 implementation assigns monotonic allocation IDs. Every allocation has its
own pageable backing and padded VA reservation; cache metadata addresses chunks as
`{allocation_id, chunk_index}`. A physical frame owns one reusable VMM handle and at
most one mapping, so allocation lifetime, logical address lifetime, and frame lifetime
remain independent.

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

### Transfer engine

The transfer engine uses a bounded pool of pinned staging memory. The complete backing
store remains pageable unless a backend proves a better choice. H2D, D2H, and compute
streams are kept distinct when the device exposes useful copy engines.

Phase 2 divides the default four pinned buffers into two H2D and two D2H staging slots.
Each slot has generation ownership: it cannot be reused until the matching event is
successfully retired and, for D2H, the bytes have been copied into pageable backing.

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

## Backend strategy

The portable core uses the CUDA Driver API. The Windows backend adds DXGI/WDDM budget
observation and LUID-based adapter matching. Host VMM is an optional capability, not a
requirement for the resident-only baseline: pageable backing plus a bounded pinned
staging pool remains the compatibility path.

All optional behavior is capability-gated at runtime. A recent header or driver version
does not imply that a particular GeForce implements every location or memory-pool mode.

## Planned public boundaries

- A stable C ABI for framework adapters and injected modules.
- A C++ implementation API hidden behind the ABI.
- Versioned JSON telemetry and capability reports.
- Explicit working-set transactions for known libraries.
- Conservative allocation-level transactions for generic interception.

`xvram-probe` validates backend capabilities and transport assumptions.
`xvram-vmm-poc` is the deliberately narrow stable-address FIFO proof.
`xvram-cache-bench` is the Phase 2 boundary: its isolated worker drives the internal
multi-allocation residency engine through deterministic workloads and returns telemetry
to a controller-owned report and trace sink. Neither internal library is a public SDK
ABI. That boundary remains Phase 3.
