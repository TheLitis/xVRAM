# Adaptive lossless compressed backing

Phase 5 makes compressed host storage an authoritative part of the production
`xvram_residency` runtime. Compression is not a fixed VRAM multiplier and is not a
lossy approximation: every representation can be recovered byte-for-byte, and the
working set of every GPU operation must still fit the live resident target.

> [!IMPORTANT]
> The implementation, benchmark, contracts, and acceptance automation belong to this
> phase. The RTX 3070 core and compression-enabled PyTorch gates are not recorded as
> complete in this document. The presence of the scripts is not acceptance evidence.

Phase 1--4 ABI and schema files remain frozen. The old Phase 2 `CacheManager` remains a
raw-backing regression implementation; Phase 5 changes the production `Runtime` used by
the SDK and lease-scoped PyTorch path rather than creating another transfer manager.

## What “elastic” means

`--logical-size auto` stays deliberately conservative:

```text
auto logical size = min(1.5 x total VRAM, raw-safe host limit)
```

An explicitly requested size has no `2x` clamp. Admission is determined at runtime by:

- CUDA virtual-address reservation capacity;
- the configured and currently safe host-store budget;
- conversion scratch, pinned staging, and raw spill credits;
- the largest declared operation working set;
- live CUDA free memory and, on Windows, WDDM local-memory budget;
- frame, nvCOMP workspace, codec-slot/status, and compute-scratch charges.

The practical ratio is therefore data- and workload-dependent. A structured model may
remain below its host cap at `3x` or `4x VRAM`; random weights may receive essentially
no compression and exhaust the safe host budget first. `capacity` mode asks the runtime
to prioritize a smaller lossless representation, but does not waive any budget or
resident-working-set invariant.

## HostBackingStore

Every logical allocation is still split into residency chunks and retains a stable,
padded CUDA VA reservation. Host storage is addressed by `ChunkKey {allocation_id,
chunk_index}`. Exactly one host representation is authoritative for each key:

| Representation | Meaning |
| --- | --- |
| `invalid` | No readable content. Used for uninitialized/proven-dead storage. |
| `implicit_zero` | Every valid byte is zero; no payload is stored. |
| `raw` | The valid tail is stored byte-for-byte. |
| `lz4_blocks` | A verified `xvram_lz4_blocks_v1` container is authoritative. |

The authoritative image is immutable. A write or conversion creates a monotonically
new generation, verifies it, and atomically commits it. The old image is retained until
commit succeeds. Generation overflow, stale-generation input, token mismatch, allocation
failure, or budget failure leaves the old authority unchanged.

On Windows, newly created full 64 KiB raw blocks use independent pageable OS
reservations rather than long-lived large allocations in the process heap. This avoids
heap-churn degradation at elastic sizes while retaining exact per-block ownership:
a partial write cannot keep an uncharged, otherwise-dead chunk-sized slab alive.
Tails and existing codec-owned payloads retain their exact-sized storage.

The host budget ledger charges four categories:

```text
authoritative storage
conversion scratch
pinned staging
raw spill credits
```

A zero SDK cap requests a platform-safe automatic limit, not a zero-byte store. The
ledger uses overflow-safe arithmetic and tracks current, peak, and rejected bytes.
SDK/PyTorch v2 resolve automatic cap and headroom from one live RAM sample inside
production setup, after CUDA context initialization. Explicit limits are checked
against that same sample; an earlier estimate is not reinterpreted as a fixed cap.
Candidate storage and its temporary raw snapshot are charged before allocation; queued
CPU inputs/outputs and codec pinned buffers also consume this budget. A spill credit is
reserved before the first write submission and retained until safe host replacement or
discard. Compressed size alone is therefore not the host-admission requirement.

### Block container v1

`xvram_lz4_blocks_v1` uses a 64 MiB residency chunk by default and independent 64 KiB
codec blocks. It stores:

- format version and `valid_bytes`;
- immutable content generation;
- a 128-bit verification token over byte values, order, block position, and valid tail;
- one tag and uncompressed size per block;
- a payload only for `raw` or `lz4` blocks.

An all-zero block is tagged `implicit_zero`. A non-zero block is encoded as a raw LZ4
block and retained as `lz4` only when the payload is smaller than its source. Otherwise
that block is stored `raw`. This per-block rule excludes expansion while keeping the
whole container CPU-decodable without nvCOMP or a CUDA device.

A full host write does not read the prior chunk. A partial host write materializes only
the intersecting blocks, applies the patch, recomputes the token, and atomically commits
the new mixed representation. A GPU-produced LZ4 candidate is fully decoded and token-
checked on the CPU before it can become authoritative.

CPU encoding uses one verified source snapshot and a bounded queue. Adjacent blocks
with equal lengths and byte-identical content reuse the predecessor's retired encoded
result, including across queue batches. Each block still stores its own payload and the
whole candidate is verified before commit. This avoids repeated encode work on
structured data without adding an unbounded dictionary or a deduplicated storage format.

## Transfer paths

The configured codec is `auto` or `lz4`; Phase 5 implements three observable paths.

### `raw`

Raw or CPU-materialized bytes use the existing bounded pinned staging path. Mapping,
`cuMemSetAccess`, copy, compute, completion, write-back, and unmap remain owned by the
production residency runtime.

### `cpu_lz4_gpu_decode`

The CPU creates/owns a verified LZ4 container. H2D transfers the compressed block
payloads and metadata, then nvCOMP's CUDA backend decodes directly into the stable
mapped VA. A verification step completes before the chunk is declared resident.

### `nvcomp_gpu_codec`

Read admission is the same event-safe GPU decode. For dirty write-back the runtime
waits for the compute generation, runs nvCOMP encode on its encode stream, retires the
codec event, transfers the compressed result to host, verifies it, and commits it before
unmapping. If the encoded result is not beneficial, the previously reserved spill is
used for raw D2H.

nvCOMP 5.3.0.16 supports the CUDA 13/Ampere execution path used by the RTX 3070 gate.
The Blackwell-only hardware Decompression Engine is not selected or required.

## Codec slots and event safety

A codec slot owns an operation ID, chunk key, source generation, monotonically increasing
slot generation, CUDA event, status buffers, and bounded workspace slice. Encode and
decode use separate runtime streams. A slot cannot be reused until:

1. its submission status is known;
2. `cuEventQuery` returns `CUDA_SUCCESS`;
3. output statuses, valid sizes, and token have been checked;
4. a D2H candidate has been atomically committed when applicable.

`CUDA_ERROR_NOT_READY` is the only non-terminal query result accepted. Any other result
poisons the operation. Before a write-capable lease first dirties a chunk, the runtime
reserves a full raw spill credit. Under host-budget pressure it first writes back eligible,
already-retired dirty frames outside the entire upcoming working set, releasing their old
authority and spill credits through the normal verified commit path. This does not unmap
those frames or relax the budget; admission is retried against the exact ledger after each
commit. If no safe reclaim can make the reservation fit, no write kernel is submitted.
Codec waits use the same deadline-checked yielding loop as ordinary runtime events.
They do not request sub-millisecond sleeps that Windows can round to a scheduler tick,
and they do not change global timer resolution or NVIDIA/TDR settings.

A failure before codec submission can take a raw/CPU fallback only while the existing
authority is still known-valid. Once submission may have happened, an event-record or
event-query error makes ownership uncertain. The runtime then quarantines the codec
slot, mapping, handle, original VA reservation, and isolated worker. It never unmaps or
reuses a resource based on a guessed completion state.

Live budget shrink stops new launches and prefetch, drains in-flight compute/codec
generations, commits dirty results, and then releases surplus codec buffers and frames.
Codec workspace, slot buffers, status/token buffers, and compute scratch reduce the
device cache target; they are not hidden allocations outside telemetry.
After codec buffers are suspended, restoration requires ten consecutive safe budget
samples; an unsafe sample resets that count. Slot creation totals may span several such
epochs, while the reported slot peak measures concurrent ownership against the configured
cap.

## Adaptive and capacity policies

The cost model keeps EWMA observations per path and content generation for:

- stored/raw ratio;
- CPU and GPU encode/decode time;
- H2D/D2H bandwidth and staging time;
- CPU queue delay and availability;
- GPU occupancy and SM opportunity cost;
- reuse count and dirty rate.

Two observations are required before a path is considered calibrated. `adaptive`
chooses a compressed path only when its predicted total time beats raw by at least:

```text
max(10% of predicted raw time, 50 microseconds)
```

Three consecutive unfavorable probes mark the current content generation
`never_compress`. That history does not leak across a content change. `capacity` chooses
the smallest valid representation among calibrated paths and may accept a time cost to
stay within the host-store budget. Incompressible blocks remain raw in either mode.

## SDK ABI v2

`<xvram/xvram.h>` and API v1 are unchanged. Compression-aware C callers include
`<xvram/xvram_v2.h>` and request `XVRAM_ABI_VERSION_2` through the existing sole export:

```c
xvram_api_v2 api = {0};
xvram_status status = xvram_get_api(
    XVRAM_ABI_VERSION_2,
    (uint32_t)sizeof(api),
    &api);

xvram_session_config_v2 config = XVRAM_SESSION_CONFIG_V2_INIT;
config.compression_mode = XVRAM_COMPRESSION_ADAPTIVE;
config.compression_codec = XVRAM_COMPRESSION_CODEC_AUTO;
config.host_store_cap_bytes = 0;       /* automatic */
config.host_headroom_bytes = 0;        /* automatic */
config.compression_workspace_cap_bytes = 256ULL << 20;
config.codec_slots = 2;
config.codec_workers = 2;
```

The v2 config embeds a complete size-tagged v1 config. The v2 API table embeds the
complete v1 table at offset zero and adds `session_create_v2` and
`session_get_telemetry_v2`. Telemetry separates logical from stored and PCIe bytes,
representations, path decisions/fallbacks, generations, spill/workspace/slot peaks, and
CPU/GPU codec time.

Requesting ABI v1 creates the original eager-raw session and never initializes a codec.
There is no extra shared-library export and no silent v1-to-v2 behavior change.

## PyTorch v2

The lease-scoped runtime keeps compression off by default:

```python
runtime = xvram.torch.InferenceRuntime(
    device=0,
    compression="adaptive",          # off | adaptive | capacity
    compression_codec="auto",       # auto | lz4
    host_store_cap="auto",
    host_headroom="auto",
    compression_scratch_cap="256MiB",
    codec_slots=2,
    codec_workers=2,
)
```

`off` loads the private ABI v1 table and continues to emit the Phase 4b v1 report/trace
and `XVT1` protocol. `adaptive` and `capacity` require the private v2 table; failure to
load it is a preflight error, not a downgrade. `_wrap_resolved_v1` and its live-view
accounting are unchanged. Elastic models use a streaming `StateProvider` so the loader
does not keep a second complete raw copy of all weights.

Compression-enabled `xvram-torch-bench` uses `xvram.pytorch_inference` schema v2,
`xvram.pytorch_trace` v2, and `XVT2`. The Phase 4b v1 artifacts remain byte-for-byte
frozen for off-mode.

## Compression benchmark

The controller-owned CLI is:

```text
xvram-compression-bench
  --device <ordinal>                       default: 0
  --logical-size <auto|size>               default: raw-safe auto
  --chunk-size <size>                      default: 64MiB
  --cache-target <auto|size>               default: auto
  --compression-policy <adaptive|capacity|both>
                                             default: adaptive
  --path <auto|raw|cpu-lz4-gpu|gpu-lz4|all>
                                             default: auto
  --codec <auto|lz4>                       default: auto
  --policy <clock|lru|both>                default: clock
  --scenario <suite|compressible-read|incompressible-read|reuse|
              dirty-writeback|mixed|budget-pressure>
                                             default: suite
  --passes <2..8>                          default: 2
  --warmup-passes <0..8>                   default: 2
  --measurement-passes <1..16>             default: 5
  --host-store-cap <auto|size>             default: auto
  --host-headroom <auto|size>              default: auto
  --device-headroom <size>                 default: 512MiB
  --compression-scratch-cap <size>         default: 256MiB
  --codec-slots <2..8>                     default: 2
  --codec-workers <1..8>                   default: 2
  --staging-slots <2..8>                   default: 4
  --prefetch-distance <0..8>               default: 2
  --budget-poll-ms <n>                     default: 100
  --stall-timeout-ms <n>                   default: 5000
  --timeout-seconds <n>                    default: 900
  --seed <hex-u64>                         default: 0x585652414D503035
  --trace <path>
  --json <path|->
  --compact-json
  --no-text
  --include-identifiers
  --version
  --help
```

`suite` covers the six named scenarios. Forced `all` expands applicable policies,
paths, and replacement policies from identical initial bytes and seed. Two warmups and
five measured passes are the controlled compressible-performance fixture.

The public process owns JSON, text, and trace files. CUDA work is isolated in a child
that speaks the length-prefixed `XVZ1` protocol: `plan`, `heartbeat`, `progress`, `trace`,
and `final`, with a 1 MiB payload limit and 64 KiB trace batches. A heartbeat is emitted
every second. Only monotonically advancing work counts as progress for the 15-second
watchdog; the overall default deadline is 900 seconds. Windows uses a kill-on-close Job
Object and Linux uses a dedicated process group. Crash, hang, malformed/truncated frame,
or output failure still produces a schema-valid controller report and reaps the worker.

The strict report has identity:

```text
schema_version: 1
report_type: "xvram.adaptive_compression"
```

Required sections are `build`, `system`, `device`, `configuration`, `workloads`,
`backing`, `codec`, `telemetry`, `proof`, `outcome`, `cleanup`, and `diagnostics`.
Optional JSONL records use `xvram.compression_trace` v1 with monotonic sequence/time,
chunk and operation identities, source/target/slot generations, representation
transition, path, logical/physical bytes, and reason. Neither contract permits CUDA
virtual addresses, pointers, or native stream handles. Identifiers are redacted by
default.

The trace validator checks causal transfer chains, not just the JSON shape. H2D must
precede decode; encode must retire before compressed D2H; each completed chain must
retire the same chunk, operation, and source/slot generation. Duplicate submissions,
reused identities, incomplete chains in completed reports, and disagreement with GPU
codec counters fail validation. PCIe counters separate payload from metadata and include
traffic from rejected compressed candidates; the total is not assumed to be bounded by
logical bytes.

Exit codes retain the project family:

| Code | Meaning |
| ---: | --- |
| `0` | Requested lossless proof completed. |
| `23` | Prerequisite, configuration, or initial working set is unsafe. |
| `24` | Corruption, checksum, or reference mismatch. |
| `25` | Host/device OOM or dynamic budget pressure. |
| `26` | Worker timeout. |
| `27` | CUDA, codec, platform, protocol, or cleanup failure. |
| `64` | CLI usage error. |
| `70` | Internal controller error. |
| `74` | Output/trace I/O error. |

## Dependencies and relocation

The build pins and verifies:

- LZ4 1.10.0 source archive;
- NVIDIA nvCOMP 5.3.0.16 CUDA 13 redistributable.

LZ4 is compiled into the residency core. nvCOMP and its companion runtime are copied
beside compression-enabled targets and installed with their license and notice. Unless
an internal caller supplies an explicit absolute path, runtime lookup uses the
application-local package. Configure requires the pinned headers and actual GPU library;
the loader verifies that file's SHA-256 before and after loading and requires the exact
runtime semantic version 5.3.0. The file pin identifies package build 5.3.0.16. Reports
record the verified library hash, not the outer wheel hash or an unchecked expected value.

The GPU library SHA-256 pins are:

| Platform | Library | SHA-256 |
| --- | --- | --- |
| Windows x64 | `nvcomp64_5.dll` | `86af6147413d09233c7ac78c968163a1b529c4e22f20c9fd8b8bab7894a29c35` |
| Linux x64 | `libnvcomp.so.5` | `fb37c8bd862dab666b1dfc2a116c2447d8c7ddebdf2a03b21747d9bc9e133a9d` |

CI includes a relocated-package load check so lookup is exercised away from the build
tree and a machine-wide nvCOMP installation.

An offline build may set `XVRAM_NVCOMP_REDIST_ROOT` to an unpacked, verified package and
set `XVRAM_FETCH_NVCOMP_REDIST=OFF`. The external GPU library must still match its
platform pin; `XVRAM_NVCOMP_EXPECTED_LIBRARY_SHA256` supplies an additional check, not
a way to replace the pin. Compression-off/no-driver tests do not require a working
NVIDIA driver. A forced GPU codec path reports a prerequisite error when its pinned
runtime or device support is unavailable.

## Reproducing the gates

Run the hardware gates sequentially on Windows WDDM with the specified RTX 3070,
PyTorch 2.13/CUDA 13 environment. From a Visual Studio developer shell, configure a
Release build with tests and warnings-as-errors, then run the local tests:

```powershell
cmake -S . -B build/phase5-release -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DXVRAM_BUILD_TESTS=ON `
  -DXVRAM_WARNINGS_AS_ERRORS=ON -DXVRAM_BUILD_TORCH_RUNTIME=ON
cmake --build build/phase5-release --parallel
ctest --test-dir build/phase5-release -C Release --output-on-failure
```

Use the Python interpreter that provides the intended PyTorch installation and install
`tests/requirements.txt`. Set CMake's `Python3_EXECUTABLE` and pass the same absolute
path through the runners' `-Python` option if `python` resolves to a different
environment. The PyTorch runner also accepts explicit `-CMake` and `-Ninja` paths when
these tools are not on `PATH`.

```powershell
scripts/run-phase5-compression-acceptance.ps1 `
  -BuildDirectory build/phase5-release -Configuration Release `
  -OutputDirectory artifacts/phase5-core -Python python

scripts/run-phase5-sdk-v2-hardware-smoke.ps1 `
  -BuildDirectory build/phase5-release -Configuration Release

scripts/run-phase5-pytorch-acceptance.ps1 `
  -BuildDirectory build/phase5-pytorch -Configuration Release `
  -OutputDirectory artifacts/phase5-pytorch -Python python
```

The SDK smoke exercises public v2 session creation, host I/O, dirty eviction, compressed
reload, telemetry, and cleanup. The PyTorch runner first runs the full compression-off
Phase 4b matrix, then the v2 matrix. Keep each gate's output directory with its reports
and traces; a successful smoke alone does not substitute for either full matrix.

Core and PyTorch gate manifests record the full Git revision, dirty-worktree status,
source-input fingerprint, and hashes of the tested executables/runtime. Reports must
match that revision, and source/binary bytes must stay unchanged during the gate.
An earlier completed manifest is removed before a rerun so a failed run cannot inherit
an old success marker. A `4x` skip is accepted only for an explicit host-memory preflight
failure, not for an unrelated device, codec, or configuration prerequisite.

An individual core report/trace pair can be checked without a GPU:

```powershell
python scripts/validate_phase5_compression_trace.py `
  --report-schema schemas/adaptive-compression-report-v1.schema.json `
  --trace-schema schemas/compression-trace-record-v1.schema.json `
  --report compression.json --trace compression.jsonl
```

## RTX 3070 acceptance — pending

`scripts/run-phase5-compression-acceptance.ps1` derives exact byte sizes from the probe,
runs the core matrix, validates every report/trace against the strict schemas, checks
semantic accounting, and verifies that no worker remains. The required matrix is:

| Scenario | Logical size | Required evidence |
| --- | ---: | --- |
| Forced path parity | `1.1x` | Raw and both LZ4 GPU paths have one digest. |
| Incompressible adaptive | `1.5x` | Calibration selects raw; no expansion is stored. |
| Compressed read/reuse | `3x` | Compressed authority and real H2D byte reduction. |
| Dirty write-back | `3x` | GPU encode precedes PCIe D2H. |
| Mixed phase change | `3x` | Raw/compressed decisions and history-driven switch. |
| Budget pressure | `2.5x` | Codec workspace is charged; shrink/grow is event-safe. |
| CLOCK/LRU parity | `3x` | Final digests match. |
| Capacity stress | `4x` | Runs only after successful live host preflight. |

Every completed run must reconcile path decisions, logical/PCIe bytes, generations,
maps/SetAccess/unmaps, and recorded/retired events; report zero unsafe remaps,
transitions, corruption, dropped trace records, and live resources; stay within host and
device scratch caps; match the reference; finish cleanup; emit no diagnostics; and reap
its worker. The controlled compressible fixture additionally requires median compressed
end-to-end time below raw. Other scenarios record performance without a threshold.

The second gate reruns the Phase 4b matrix in compression-off mode and adds the v2
Llama-like/structured-model cases, CLOCK/LRU parity, prefetch distance zero/two, and
sequence-128 scratch/attention smoke. Do not update the roadmap to “complete” until both
hardware gates and the full GitHub Actions matrix actually pass.
