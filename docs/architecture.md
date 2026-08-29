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

### Residency manager

The manager is the correctness authority. It resolves a requested working set into
physical mappings, schedules dirty write-back, waits on the necessary events, and only
then authorizes launch. It also reacts to WDDM budget changes by shrinking its target
cache size.

### Transfer engine

The transfer engine uses a bounded pool of pinned staging memory. The complete backing
store remains pageable unless a backend proves a better choice. H2D, D2H, and compute
streams are kept distinct when the device exposes useful copy engines.

### Policy

Policy may combine explicit next-use distance, hotness, reuse count, cost to reload,
dirty state, and allocation class. Policy selects victims but cannot violate pinning,
in-flight, or working-set constraints.

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
`xvram-vmm-poc` is the next deliberately narrow boundary: an isolated controller and
worker prove stable-address VMM reuse with pageable backing and a fixed FIFO schedule.
The PoC's internal executor library is not a public SDK ABI and must not be confused
with the Phase 2 allocator or residency manager.
