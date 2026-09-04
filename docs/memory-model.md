# Memory model

## Tiers

System RAM is the capacity and canonical backing tier. VRAM is the resident execution
tier. In strict mode, kernels do not directly dereference cold host-backed ranges.
Canonical does not mean permanently materialized as raw pages: the Phase 5 production
runtime may keep a chunk as an implicit zero or verified lossless LZ4 container.

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

The residency state and host representation are intentionally separate. A clean
resident chunk can be backed by any valid representation; a dirty resident chunk has a
newer device generation whose eventual host replacement is protected by a pre-reserved
raw spill credit. The authoritative host representation is one of:

```text
invalid        no readable content; legal only for proven-dead or provisional storage
implicit_zero  all valid bytes are zero and consume no payload
raw            valid bytes are stored uncompressed
lz4_blocks     CPU-decodable xvram_lz4_blocks_v1 generation
```

Every successful write, compression, decompression/materialization, or GPU-produced
commit creates a monotonically newer immutable generation. A conversion never mutates
the current authority in place. Its replacement becomes visible only after sizes,
tail bytes, generation, and the 128-bit content token have all been verified.

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
8. Runtime-owned CUDA-library workspace is bounded and charged against the same live
   device target and headroom as cache frames.
9. A borrowed device address is valid only inside the transaction generation that
   resolved and pinned it.
10. Each logical chunk has at most one authoritative host generation; a candidate does
    not replace it until complete verification and atomic commit.
11. Every first-dirty write-capable chunk has a raw spill credit reserved before GPU
    submission, so an incompressible result cannot destroy the last valid authority.
12. Codec slots, pinned staging slots, mappings, and event generations cannot be reused
    before successful event retirement and required host commit.
13. Only `CUDA_SUCCESS` and `CUDA_ERROR_NOT_READY` are valid polling outcomes. Unknown
    post-submission state quarantines the affected resources and worker.
14. Host authoritative bytes, conversion scratch, pinned buffers, and spill credits are
    charged independently from the device cache/workspace budget.

## Chunk sizing

CUDA VMM minimum and recommended allocation granularities are queried per device and
location. The runtime chunk size is a policy choice aligned to the required
granularity. It must balance mapping overhead, internal fragmentation, scheduling
precision, and transfer efficiency. No fixed page size is assumed globally.

The Phase 5 container block is distinct from the residency chunk and fixed by
`xvram_lz4_blocks_v1` at 64 KiB. For the default 64 MiB chunk there are 1024 independent
blocks. Each block records its uncompressed length and is tagged `implicit_zero`, `raw`,
or `lz4`. The final block carries only the logical allocation's valid tail bytes. An LZ4
payload is committed only when it is smaller than that block's raw bytes; mixed raw and
compressed blocks are therefore normal and prevent expansion.

The CPU encoder may reuse the encoded result of an immediately preceding, byte-identical
block after that result retires. Each block still stores its own payload. This is a codec
work optimization, not additional capacity from deduplication; the verified source
snapshot, queued jobs, candidate payloads, and spill remain charged to the host ledger.

## Access modes

Working-set declarations classify each range as read-only, read-write, or write-only.
Read-only clean data can be discarded from VRAM without D2H traffic. Read-write data
becomes dirty at the operation boundary. Fine-grained dirty tracking is a later
optimization and cannot weaken the conservative boundary rule.

Phase 2 normalizes overlapping byte ranges and merges their access strength. A
full-chunk write-only declaration can omit H2D; a partial write-only declaration must
first preserve untouched bytes through H2D.

The same rule applies to host backing. A complete host write creates a new image without
reading the old generation. A partial host write decodes or materializes only the
intersecting 64 KiB blocks and retains untouched immutable blocks. Cancellation of a
provisional full write-only mapping invalidates that provisional content rather than
manufacturing a cache hit. `discard_dead` deletes backing only after the proven last-use
event retires; ordinary release writes dirty data back.

Phase 3 applies these modes to strided matrices rather than conservatively declaring a
single dense bounding box. For each GEMM tile, A and B segments are read-only. C is
read-write when beta or an earlier K panel contributes to it. With beta equal to zero,
fully covered C chunks are write-only; boundary chunks still preserve untouched host
bytes. The union of these exact segments is normalized before the transaction is
admitted.

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

The public ABI does not turn a stable reservation into a permanently dereferenceable
pointer. Generic callbacks receive resolved addresses only after their declared ranges
are resident and pinned. They may enqueue work on the supplied stream and must forget
those values on return. The runtime's post-callback event is the generation boundary
that later permits write-back, eviction, or remapping.

GEMM plans preserve the same rule across M/N/K tiling. K panels for one C tile are
contiguous. Each panel is an event-safe transaction, so C may remain resident or may be
written back and loaded again without changing the accumulation result; A/B panels may
be prefetched or evicted between transactions. cuBLAS workspace comes from a separate
reusable pool, cannot alias a physical frame or logical allocation, and cannot be reused
until the library event generation completes.

## Compressed transfer generations

The CPU LZ4 path can always reconstruct raw bytes without a CUDA device. Compressed GPU
transfer adds another event-fenced generation rather than weakening that fallback:

```text
H2D decode
  select verified host generation
    -> acquire codec slot(source_generation, slot_generation)
    -> map full residency chunk + SetAccess
    -> copy compressed blocks to device input
    -> nvCOMP CUDA decode into stable VA
    -> verify content token and valid tail
    -> retire event
    -> resident_clean

D2H encode
  retire compute generation
    -> reserve/retain raw spill credit
    -> nvCOMP CUDA encode resident bytes
    -> retire encode event
    -> copy compressed payload and statuses to host
    -> CPU-decode and verify candidate
    -> atomic host-generation commit
    -> unmap/reuse
```

If compression is not beneficial, D2H uses the reserved raw spill. A failure before
codec submission may choose raw or CPU fallback while the old host authority remains
valid. A record/query failure or any ambiguity after submission quarantines the slot,
mapping, physical handle, VA reservation, and worker; cleanup reports that uncertainty
instead of freeing an address that might still be in use.

Trace identities include the chunk, operation, source generation, and staging or codec
slot generation. Completed traces must show matching submission/retirement chains;
encode retirement precedes compressed D2H submission. Reported physical PCIe bytes
include payload plus metadata, so total PCIe bytes need not be smaller than logical
bytes on every transfer. Payload, metadata, and rejected compression-candidate traffic
are accounted for separately.

## Capacity and sizing

Compression changes the host representation, not the resident-only execution rule:

```text
WorkingSet(operation) <= live VRAM/WDDM target after codec and compute workspace
```

Logical size has no fixed `2x` or `4x` multiplier cap. Automatic sizing remains
conservative and raw-safe at `min(1.5 x total VRAM, safe host limit)`. An explicit size
is admitted only while all of the following remain true:

- CUDA VA can reserve every padded logical allocation;
- the authoritative store plus conversion scratch, pinned staging, and required spill
  credits fit the live host budget and headroom;
- the maximum declared working set fits the live device target;
- frames, nvCOMP workspace, codec slots/status buffers, compute scratch, and device
  headroom fit current CUDA and WDDM observations.

Consequently a compressible workload may run beyond `2x VRAM`, while an incompressible
workload can reach host-budget OOM at a smaller ratio. Capacity mode changes selection
policy, not these safety constraints, and there is no disk spill in Phase 5.
