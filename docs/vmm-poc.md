# Explicit CUDA VMM proof of concept

`xvram-vmm-poc` is the Phase 1 proof that xVRAM can process a logical `uint32`
array larger than the selected GPU's total VRAM while every kernel touches only a
bounded, explicitly resident physical window. It is a validation executable, not a
transparent allocator or a production runtime.

## What the proof establishes

The complete logical array lives in pageable system RAM. Each logical tile has a fixed
address inside one reserved CUDA VA range. A slot owns one reusable CUDA VMM allocation
handle and one pinned staging buffer; as the FIFO advances, that handle is mapped at the
next logical tile's stable address. The default pipeline has two 64 MiB slots, so only
128 MiB of device memory and 128 MiB of pinned host memory are used for a logical array
that is normally much larger.

For every mapping the worker calls `cuMemSetAccess` again. A slot cannot be verified,
unmapped, or reused until `cuEventQuery(slot_done)` reports `CUDA_SUCCESS`;
`CUDA_ERROR_NOT_READY` is the only retryable result. The final partial tile is mapped at
the full VMM granularity, while copies and the kernel are limited to its valid bytes.
These rules follow the CUDA Driver API contracts for
[virtual memory management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
and [events](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html).

The embedded `xvram_vmm_transform_v1` PTX kernel applies a deterministic in-place
transform using the absolute 64-bit element index, pass number, and seed. Passes
alternate forward and reverse traversal. Every returned tile is compared word-for-word
with the CPU transform before write-back. Each mode then performs another complete CPU
comparison and produces a 128-bit digest. Reference and pipeline modes start from the
same regenerated backing data and must finish with identical digests.

The pipeline is intentionally a fixed FIFO proof, not a residency cache. Hotness,
arbitrary prefetch, dirty-state policy, and eviction choice belong to Phase 2.

## Safety preflight

The executable refuses to run unless all of these conditions are true:

- CUDA VMM and unified virtual addressing are supported;
- logical bytes are strictly greater than total device VRAM;
- the resident window plus configured device headroom fits CUDA free memory;
- on Windows, the same amount fits the current DXGI/WDDM local-memory budget;
- pageable host backing, pinned staging, and service headroom fit the conservative host
  memory limit.

Automatic logical size is the smaller of `1.5 * total VRAM` and the safe host limit.
The host limit retains at least the larger of 4 GiB and 25% of physical RAM, plus pinned
staging and a service reserve. Requested chunk size is rounded up to a common multiple
of the device's minimum and recommended VMM granularities.

Windows rechecks the WDDM budget every 32 retired tiles. Budget pressure stops new
launches, drains submitted work, and returns exit code 25. A kernel tile longer than
250 ms prevents another kernel launch.

## Controller isolation

The public process is a controller. It launches the same executable in a hidden worker
mode, places it in a Windows Job Object or a Linux process group, and exchanges `XVP1`
length-prefixed messages. The protocol accepts payloads up to 1 MiB and carries plan,
progress, and final-report frames.

The worker reports every retired tile and emits a heartbeat during long CPU-only scans.
The controller terminates and reaps only the worker after 15 seconds without progress or
after the overall deadline (120 seconds by default). It still writes a schema-valid
timeout report with unknown cleanup state. The proof never edits TDR registry values or
NVIDIA system settings.

## Running the proof

The default command selects device 0, automatic logical size, 64 MiB chunks, two slots,
two passes, and both modes:

```powershell
build\vs\Release\xvram-vmm-poc.exe --json xvram-vmm-poc.json
```

An explicit hardware acceptance run looks like this:

```powershell
build\vs\Release\xvram-vmm-poc.exe `
  --logical-size 12GiB `
  --chunk-size 64MiB `
  --window-slots 2 `
  --passes 2 `
  --mode both `
  --json acceptance-12g.json
```

Size suffixes are case-insensitive binary units (`KiB`, `MiB`, `GiB`); a bare
integer is bytes. `--window-slots 1` is accepted only for reference mode. Use
`xvram-vmm-poc --help` for the complete option list.

## Report and exit contract

The strict report has `schema_version: 1` and
`report_type: "xvram.vmm_poc"`. Its schema is
[`schemas/vmm-poc-report-v1.schema.json`](../schemas/vmm-poc-report-v1.schema.json).
It records the effective memory plan, VMM granularities, embedded PTX version and
SHA-256, mapping/access/event/reuse counts, stage timings, WDDM budget observations,
digests, mismatches, proof flags, and a complete cleanup ledger. CUDA virtual addresses
are never serialized. Stable UUID/LUID/PCI identifiers remain redacted unless
`--include-identifiers` is supplied.

Exit codes are:

| Code | Meaning |
| ---: | --- |
| 0 | The complete oversubscription proof passed. |
| 23 | A prerequisite or safe configuration could not be satisfied. |
| 24 | GPU output or a mode digest did not match the CPU reference. |
| 25 | Host/device allocation failed or WDDM budget pressure was observed. |
| 26 | The isolated worker exceeded a progress or overall deadline. |
| 27 | CUDA, platform, worker protocol, or cleanup failed. |
| 64 | Command-line usage error. |
| 70 | Internal invariant failure. |
| 74 | Report output I/O failed. |

Exit code 0 additionally requires logical bytes greater than total VRAM, a physical
window smaller than the logical data, actual handle reuse, stable logical addresses,
one completed event boundary per unmap, zero unsafe remaps, complete CPU agreement, and
fully successful cleanup. Pipeline speedup is reported but is never a pass criterion.

## Failure containment

Cleanup attempts all independent stages even after an earlier error: drain events,
remove mappings, release handles, release the exact original reservation, destroy CUDA
objects, release pinned buffers, and destroy the isolated context. A failed unmap
quarantines the reservation. The handle is still released, but the reservation is not;
the worker process then acts as the final quarantine boundary and exits. A stream that
cannot be drained similarly prevents unsafe host-buffer release before context teardown.
