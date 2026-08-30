# Event-safe residency cache benchmark

`xvram-cache-bench` is the Phase 2 validation tool for the internal
`xvram_residency` engine. It is not a public allocator or SDK ABI. The benchmark proves
that a deterministic logical heap backed by pageable RAM can be processed through a
bounded, dynamically managed VRAM write-back cache without unsafe remapping.

## Runtime boundary

The controller starts the same executable in a hidden worker mode, owns report and trace
files, and exchanges length-prefixed `XVC1` frames. Protocol payloads are limited to
1 MiB and trace batches to 64 KiB. The versioned frame types are `plan`, `progress`,
`trace`, and `final`. The worker sends progress after every retired operation and a
heartbeat at least once per second during longer CPU-side work.

The controller has a 15-second no-progress watchdog and an overall deadline of 300
seconds by default. On timeout it terminates and reaps only the worker, then emits a
schema-valid report with exit code 26 and unknown cleanup fields. Windows workers are
contained in a Job Object; Linux workers use a separate process group. No TDR, registry,
WDDM, or NVIDIA setting is changed.

## Storage and residency

The canonical data for each logical allocation remains in pageable RAM. The allocation
owns a stable, padded CUDA VA reservation, while reusable physical VMM handles are
managed as frames. A frame has at most one mapping. The default 64 MiB requested chunk
is rounded up to a common multiple of the device's minimum and recommended VMM
granularities.

Four pinned buffers are allocated by default: two H2D and two D2H. The staging pool is
bounded independently from logical heap size and cache target. Each event and staging
slot has a generation that must retire before reuse, preventing ABA-style reuse.

The default logical size is the smaller of `1.5 * total VRAM` and the conservative host
limit. The default cache cap is total VRAM, but the initial and live targets are reduced
by CUDA free memory, Windows WDDM availability, configured headroom, and declared
working-set size. An initial working set that cannot fit returns 23. A later budget
shrink below the next working set returns 25 after safely draining submitted work.

## Transactions, policy, and prefetch

Access ranges are normalized into read, read-write, and write-only spans. Full-chunk
write-only operations skip H2D. Partial write-only spans first load host data to preserve
untouched bytes. At most one compute transaction executes at a time; transfer streams
can overlap demand load, speculative prefetch, write-back, and compute when the hardware
permits it.

Cost-aware CLOCK is the default eviction policy. It marks demand accesses referenced,
inserts speculative chunks cold, prefers clean victims, and deprioritizes sequential
one-touch data. LRU is a deterministic baseline. Both use `ChunkKey` as a stable final
ordering and neither can override the manager's pin, flight, alias, or working-set
checks.

Sequential prefetch defaults to distance 2 and can be disabled with distance 0.
Speculative requests are bounded by available H2D staging. Demand promotes a matching
prefetch; stale queued requests are cancelled without reusing an active generation.

## Workloads

The deterministic seed defaults to `0x585652414D503032`. `suite` restores identical
backing before each of these five cases:

| Scenario | Access pattern |
| --- | --- |
| `sequential` | Alternating forward/reverse read-write passes. |
| `reuse` | Warmup, then seven read-only cycles over a hot set equal to 50% of the initial target. |
| `random` | `4 * logical_chunk_count` deterministic uniform operations, split equally between read and read-write. |
| `read-only` | Full forward/reverse scan with no dirty transition. |
| `write-heavy` | Shuffled passes with 20% read, 60% read-write, and 20% full-chunk write-only. |
| `budget-pressure` | Warmup, controlled non-cache allocation, observed shrink, release, and hysteretic grow. |

`--policy both` regenerates identical backing and access traces for CLOCK and LRU.
Performance and prefetch metrics are observations, not pass thresholds. Correctness,
event safety, accounting, and cleanup are pass criteria.

## Command line

```text
xvram-cache-bench
  --device <ordinal>                       default: 0
  --logical-size <auto|size>               default: auto
  --chunk-size <size>                      default: 64MiB
  --cache-target <auto|size>               default: auto
  --staging-slots <2..8>                   default: 4
  --policy <clock|lru|both>                default: clock
  --prefetch-distance <0..8>               default: 2
  --scenario <suite|sequential|reuse|random|read-only|write-heavy|budget-pressure>
                                             default: suite
  --passes <2..8>                          default: 2
  --pressure-size <auto|size>              default: auto
  --device-headroom <size>                 default: 512MiB
  --budget-poll-ms <n>                     default: 100
  --stall-timeout-ms <n>                   default: 5000
  --timeout-seconds <n>                    default: 300
  --seed <hex-u64>                         default: 0x585652414D503032
  --trace <path>
  --json <path|->
  --compact-json
  --no-text
  --include-identifiers
  --version
  --help
```

Size suffixes are case-insensitive binary units (`KiB`, `MiB`, `GiB`); a bare integer
is bytes. Logical size must be `uint32` aligned. `--trace` requires a file path because
stdout is reserved for report output. Report and trace paths must differ.

Example:

```powershell
build\vs\Release\xvram-cache-bench.exe `
  --logical-size 12GiB `
  --chunk-size 64MiB `
  --policy both `
  --scenario suite `
  --passes 2 `
  --trace cache-suite.jsonl `
  --json cache-suite.json
```

## Report and trace contracts

The strict report schema is
[`xvram.residency_cache` v1](../schemas/residency-cache-report-v1.schema.json). Its
required top-level sections are `build`, `system`, `device`, `configuration`,
`workloads`, `cache`, `telemetry`, `proof`, `outcome`, `cleanup`, and `diagnostics`.
It includes:

- effective logical, chunk, cache, staging, and budget sizes;
- workload operations, access classes, cache hits/misses, transfers, evictions,
  write-backs, prefetch outcomes, timings, digests, and mismatch location;
- frame create/release/reuse, map/access/unmap, event boundary, shrink/grow, and unsafe
  counters;
- embedded PTX version and SHA-256;
- proof and cleanup ledgers.

An optional trace is newline-delimited
[`xvram.residency_trace` v1](../schemas/residency-trace-record-v1.schema.json). Each
record contains a sequence number, monotonic timestamp, event/transition, optional
allocation, chunk, operation and transaction IDs, byte count, reason, policy,
speculative flag, and event generation.

The controller, never the CUDA worker, opens and closes JSON, text, and trace sinks. A
trace-sink failure produces exit 74 and a report that marks the trace incomplete; a
partial JSONL file is never represented as a complete trace.

Raw CUDA virtual addresses are forbidden in both formats. UUID, LUID, and PCI bus ID are
redacted unless `--include-identifiers` is explicitly supplied. A completed report may
have `logical_data_exceeds_vram: false` for the intentional `0.8x` hardware case; all
oversubscribed cases must set it true. The cache must remain smaller than logical data
in every accepted gate case.

The Phase 0 capability v1/v2 and Phase 1 VMM PoC v1 schemas remain frozen. CI protects
their exact SHA-256 contracts in addition to validating new residency fixtures.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | The selected cache proof completed. |
| 23 | A prerequisite, safe configuration, or initial working set is impossible. |
| 24 | GPU output or a digest differs from the CPU reference. |
| 25 | Host/device OOM or dynamic budget pressure prevents safe progress. |
| 26 | The worker exceeded its progress or overall deadline. |
| 27 | CUDA, platform, protocol, worker, or cleanup failed. |
| 64 | Command-line usage error. |
| 70 | Internal invariant failure. |
| 74 | Report or trace output I/O failed. |

## RTX 3070 acceptance

Build Release with tests enabled, install `tests/requirements.txt`, then run:

```powershell
scripts\run-phase2-rtx3070-acceptance.ps1 `
  -BuildDirectory build\vs `
  -Configuration Release `
  -OutputDirectory artifacts\phase2-rtx3070
```

The script obtains exact total VRAM from `xvram-probe`, runs suite at `0.8x`, `1.1x`,
`1.5x`, and `2.0x` for CLOCK and LRU, and runs CLOCK budget-pressure at `1.5x`. It
validates every report against the strict schema, enforces semantic counter/proof/
cleanup rules, compares policy digests, and rejects a residual benchmark process.
