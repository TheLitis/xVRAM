# Memory model

## Tiers

System RAM is the capacity and canonical backing tier. VRAM is the resident execution
tier. In strict mode, kernels do not directly dereference cold host-backed ranges.

Each logical chunk has one authoritative state and may have a valid cached copy. The
implemented Phase 2 state machine is:

```text
host_clean
  -> prefetch_queued
  -> mapping
  -> h2d_in_flight
  -> resident_clean
  -> resident_dirty
  -> writeback_queued
  -> d2h_in_flight
  -> resident_clean
  -> evicting
  -> host_clean

any state -> poisoned
```

Pin count, physical-frame ownership, staging ownership, and event generation are
orthogonal to this state. Dirty is established immediately after successful submission
of a write kernel, rather than after its later retirement.

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

Phase 2 normalizes overlapping byte ranges and merges their access strength. A
full-chunk write-only declaration can omit H2D; a partial write-only declaration must
first preserve untouched bytes through H2D.

## Address stability

Logical addresses are reserved for the lifetime promised to the integration layer.
Physical handles may change underneath that reservation. Consumers must not launch work
against an address without participating in the working-set transaction.

Phase 1 demonstrates this invariant with a padded CUDA VA reservation whose aligned
logical base remains fixed. Every logical tile therefore keeps the address
`logical_base + tile_offset`. Each slot carries a reusable physical handle between
those tile addresses, reapplies access rights after every mapping, executes, copies
back, waits for the completion event, and unmaps the full chunk. The final partial
chunk changes only the valid copy and kernel byte count, never the mapped range.

Phase 2 extends the invariant to multiple allocations. Each allocation releases the
exact original padded reservation, not merely its aligned logical base. Physical frame
handles move between chunk addresses only after all users and the relevant event
generation have retired. `cuMemSetAccess` is applied after every new mapping, and unmap
always covers the original mapped chunk range.
