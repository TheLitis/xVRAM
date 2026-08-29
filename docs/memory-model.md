# Memory model

## Tiers

System RAM is the capacity and canonical backing tier. VRAM is the resident execution
tier. In strict mode, kernels do not directly dereference cold host-backed ranges.

Each logical chunk has one authoritative state and may have a valid cached copy:

```text
HOST_CLEAN -> PREFETCHING -> VRAM_CLEAN -> VRAM_DIRTY
     ^                                          |
     +---------------- EVICTING <---------------+
```

Additional orthogonal flags describe whether a chunk is in flight, locked into the
next working set, compressed in the host tier, or undergoing codec work.

## Core invariants

1. A mapped logical range has exactly one physical VRAM owner.
2. An unmapped or remapped range has no possible in-flight GPU access.
3. Dirty VRAM data is written back before its physical pages are reused.
4. A kernel launch is authorized only after its declared ranges are resident and its
   dependency events are satisfied.
5. The physical target stays below the current WDDM budget minus configured headroom.
6. Pinned host memory is bounded independently from the logical heap capacity.
7. A failed prefetch, mapping, or eviction fails the operation; it never degrades into
   an access to an invalid address.

## Chunk sizing

CUDA VMM minimum and recommended allocation granularities are queried per device and
location. The runtime chunk size will be a policy choice aligned to the required
granularity. It must balance mapping overhead, internal fragmentation, scheduling
precision, and transfer efficiency. No fixed page size is assumed globally.

## Access modes

Working-set declarations classify each range as read-only, read-write, or write-only.
Read-only clean data can be discarded from VRAM without D2H traffic. Read-write data
becomes dirty at the operation boundary. Fine-grained dirty tracking is a later
optimization and cannot weaken the conservative boundary rule.

## Address stability

Logical addresses are reserved for the lifetime promised to the integration layer.
Physical handles may change underneath that reservation. Consumers must not launch work
against an address without participating in the working-set transaction.
