# Lease-scoped PyTorch inference

Phase 4b connects static PyTorch inference graphs to the pageable xVRAM residency
cache. Unlike the resident-only Phase 4a `MemPool`, model state and managed
activations have pageable canonical backing and a stable CUDA virtual address. A CUDA
tensor view exists only while an explicit compute lease pins every chunk the submitted
ATen region can touch.

> [!IMPORTANT]
> The lease runtime, strict planner, Python frontends, native Stable-ABI bridge,
> controller/report contract, and hardware-gate script are implemented. As of
> 2026-09-02, all six RTX 3070 scenarios validate locally. The implementation and local
> hardware gate are complete; final Phase 4b delivery still awaits the full GitHub
> Actions result.

The feature is intentionally narrow: static-shape forward inference on one CUDA device.
There is no eager fallback. Autograd, graph breaks, CUDA Graph capture, RNG, arbitrary
custom operators, distributed/NCCL execution, dynamic shapes, hidden streams, and
unproven mutation or aliasing fail preflight before a managed kernel is launched.

Phase 4b is a private integration boundary. It neither changes the public Phase 3 C ABI
nor extends the Phase 4a allocator ABI. Applications should use `xvram.torch`; the
size-tagged native control table is not a supported SDK interface.

## Safety boundary

The native residency runtime attaches to PyTorch's current primary CUDA context and
owns its compute stream, stable reservations, mappings, physical frames, events,
staging, and pageable backing. Python receives opaque session/allocation/lease
identifiers; it never receives or serializes a CUDA virtual address or native stream
handle.

One region follows this sequence:

```text
normalize declared ranges
        |
map + SetAccess + demand H2D
        |
pin chunks / begin event generation
        |
return lease-scoped views + runtime ExternalStream
        |
enqueue ATen region, drop every view
        |
seal: record completion event, mark writes dirty
        |
query event -> retire generation -> unpin
        |
prefetch future read-only state / discard proven-dead activation
```

Only `CUDA_SUCCESS` and `CUDA_ERROR_NOT_READY` are accepted from completion-event
polling. An event-record/query error, a stale lease, a live tensor view at seal, a
context/stream mismatch, or uncertainty about whether work was submitted quarantines
the worker. A mapping is never removed or a frame reused until all users of its event
generation have retired. Every new mapping is followed by `cuMemSetAccess`, and unmap
always covers the original complete mapped chunk.

The runtime supplies the stream wrapped by `torch.cuda.ExternalStream`; callers cannot
substitute a stream. The Python session outlives the wrapper. CUDA Graph capture is
rejected. A region that exceeds 250 ms retires safely but prevents another launch.
Only one compute lease may be active in a session.

## Stable-ABI bridge

`xvram_torch_runtime` is separate from `xvram_torch_allocator`. It exposes one internal
size-tagged control-table getter and the bounded scratch allocator callbacks. Its
`xvram_internal::_wrap_resolved_v1` operator creates a non-owning CUDA tensor with
`torch::stable::from_blob`. The tensor deleter only releases the bridge view ticket; the
residency runtime remains the sole storage owner.

Every view increments a native lease counter. Seal requires that counter to be zero.
The bridge is compiled with `TORCH_TARGET_VERSION` for PyTorch 2.11. The dedicated
Windows/Linux CI job builds and loads it with pinned CPU PyTorch 2.11, then upgrades the
runtime to CPU PyTorch 2.13 and loads the unchanged bridge again. The completed hardware
acceptance target is PyTorch 2.13 with CUDA 13.0.

Temporary results for operators without an `.out` form use the resident scratch pool.
The default cap is 512 MiB and is charged against the live cache target. The accepted
SDPA backend is `MATH`; `FLASH_ATTENTION` is an explicit, observable choice and is never
selected as a silent fallback.

## Graph capture and planning

`torch.export.export(model, example_inputs, strict=True)` is the canonical capture. The
custom `torch.compile` backend is anchored to that already-compiled plan. It compares
Dynamo's normalized operator targets plus output shapes/dtypes and verifies the lifted
state/user-input identities; any mismatch is rejected. The reported backend hash is the
canonical export hash after that validation, not an independently computed Dynamo hash.

The planner represents state, inputs, outputs, and temporaries as managed values with
an allocation/generation, byte range, static shape/strides/dtype, and alias group. It:

- deduplicates tied parameter storage;
- keeps view/reshape/transpose/permute/slice/unsqueeze/getitem as metadata aliases;
- merges access ranges per allocation and rejects overlapping or negative write strides;
- colors non-overlapping activation lifetimes into reusable slots;
- declares only the embedding rows/chunks selected by CPU token IDs;
- preplans all material outputs and bounded scratch before the first launch;
- discards a dead activation only after its last-use event retires;
- prefetches a bounded number of future read-only weight ranges.

The initial operator allowlist covers embedding, bias-free linear and matrix products,
elementwise add/mul/neg/SiLU, cat, explicit casts/copies, the metadata aliases above,
RMSNorm, and scaled-dot-product attention. When the declared ranges of `mm` or a
bias-free `linear` exceed the live target, the backend calls the private synchronous
tiled-GEMM hook. That hook uses the common xVRAM GEMM planner/executor, binds
cuBLAS/cuBLASLt to the runtime stream for each tile, and returns only after all tile
events retire. It supports `beta = 0` only. An oversized `addmm`, `bmm`, biased linear,
or any other oversized operator is rejected instead of acquiring an unsafe
full-working-set lease.

Activation slots that may become an `mm`/`linear` destination are conservatively kept
valid rather than liveness-discarded. A tiled `beta = 0` output is produced one tile at a
time; retaining that backing prevents a partial output tile from depending on invalid
host bytes outside the tile. Other proven-dead activation storage is discarded only
after its last-use event retires.

`StateProvider` supports meta-device modules and chunked state generation/loading so a
second complete CPU copy of model weights is unnecessary. A normal CPU `nn.Module` is
handled by the built-in module-state provider. CPU inputs are copied into pageable
backing. Results are CPU tensors by default; CUDA materialization is an explicit copy to
ordinary PyTorch-owned storage and is included in the headroom calculation.

## Python API

```python
import xvram

runtime = xvram.torch.InferenceRuntime(
    device=0,
    cache_target="auto",
    chunk_size="64MiB",
    device_headroom="512MiB",
    scratch_cap="512MiB",
    prefetch_distance=2,
)

compiled = runtime.compile_inference(model, example_inputs)
cpu_output = compiled(*cpu_inputs)
runtime.close()
```

The equivalent strict `torch.compile` path is:

```python
optimized = torch.compile(
    model,
    backend=runtime.backend(model, example_inputs),
    fullgraph=True,
    dynamic=False,
)
```

The runtime owns one compiled graph and must remain on its creating thread. It is also a
context manager. `close()` first retires an active operation, then releases logical
allocations and asks the native session to drain and tear down mappings, frames,
reservations, events, streams, staging, cuBLAS resources, and the attached context
reference. A close error is reported as cleanup failure; it is never converted into a
successful proof. A quarantined worker is a process boundary and is not reused.

## Build and load

Install the Python package into the interpreter whose PyTorch headers/libraries CMake
should use, then enable the optional private bridge explicitly:

```powershell
python -m pip install .\python
cmake -S . -B build\vs -A x64 `
  -DXVRAM_BUILD_TESTS=ON -DXVRAM_BUILD_TORCH_RUNTIME=ON
cmake --build build\vs --config Release `
  --target xvram_torch_runtime xvram-torch-runtime-control-tests
$env:XVRAM_TORCH_RUNTIME_LIBRARY = `
  (Resolve-Path .\build\vs\Release\xvram_torch_runtime.dll).Path
```

With the default `XVRAM_BUILD_TORCH_RUNTIME=AUTO`, configuration skips this private
target when a compatible PyTorch installation cannot be found. `ON` makes missing
PyTorch 2.11+ Stable-ABI headers/libraries a configure error. Linux uses
`libxvram_torch_runtime.so`; alternatively pass its absolute path as
`InferenceRuntime(library_path=...)`.

## Benchmark and report

The installed console command and module entry point are equivalent:

```text
xvram-torch-bench
  --device <ordinal>                       default: 0
  --model <llama2-like|operator-smoke>     default: llama2-like
  --model-ratio <positive number>          default: 1.5
  --layers <n>                             default: 31
  --batch <n>                              default: 1
  --sequence <n>                           default: 32
  --hidden <n>                             default: 4096
  --intermediate <n>                       default: 11008
  --heads <n>                              default: 32
  --dtype <float16|bfloat16|float32>       default: float16
  --policy <clock|lru>                     default: clock
  --cache-target <auto|size>               default: auto
  --chunk-size <size>                      default: 64MiB
  --device-headroom <size>                 default: 512MiB
  --scratch-cap <size>                     default: 512MiB
  --prefetch-distance <0..8>               default: 2
  --sdpa-backend <math|flash_attention>    default: math
  --seed <u64>                             default: 0x585652414D503034
  --timeout-seconds <n>                    default: 900
  --trace <path|->
  --json <path|->                         default: -
  --compact-json
  --no-text
  --include-identifiers
  --version
  --help
```

`--layers` and the explicit geometry determine the actual model size.
`--model-ratio` records the intended acceptance ratio; it does not resize the model.
Oversubscription is decided from the report's observed `model.logical_bytes` and
`device.total_vram_bytes`, never from that label.

Example:

```powershell
$env:PYTHONPATH = (Resolve-Path .\python).Path
python -m xvram.torch_bench `
  --layers 31 --model-ratio 1.522 --policy clock `
  --prefetch-distance 2 --sdpa-backend math `
  --trace torch-trace.jsonl --json torch-report.json
```

The public controller owns text, JSON, and trace output. Its isolated worker speaks the
length-prefixed `XVT1` protocol with a 1 MiB frame limit and a 64 KiB trace-batch limit.
The worker sends a heartbeat every second and progress for state, prefetch, discard,
region-retirement, output, and reference-verification activity. Only strictly
increasing progress resets the 15-second stall deadline; heartbeats demonstrate liveness
but do not hide a worker that stopped retiring work. The overall deadline defaults to
900 seconds. Windows uses a kill-on-close Job Object; Linux uses a dedicated process
group. A crash, malformed/oversized/truncated frame, or timeout still produces a
schema-valid report and the controller terminates and reaps the worker.

The strict `xvram.pytorch_inference` v1 report is validated by
`schemas/pytorch-inference-report-v1.schema.json`. Optional JSONL records use
`xvram.pytorch_trace` v1. Neither contract permits raw CUDA virtual addresses or native
stream handles. Exit codes use the established project family: `0`, `23`–`27`, `64`,
`70`, and `74`.

Exit `0` requires equal canonical/backend graph hashes, a fully supported graph,
complete region/tile/lease/event reconciliation, stable non-aliased addresses,
`maps == SetAccess == unmaps`, bounded residency and scratch, complete prefetch and
trace accounting, reference digest equality, zero final live views, successful cleanup,
an empty diagnostics list, and a reaped worker. When observed logical bytes exceed
physical VRAM, success additionally requires actual eviction and frame reuse; a
sub-VRAM run may succeed without claiming oversubscription.

The v1 `zero_weight_writeback` field is based on the planner proving all persistent
state accesses read-only; native D2H is currently reported as a global cache counter,
not attributed per allocation. Likewise, the resource-specific cleanup flags combine a
successful native close with allocation/mapping/event reconciliation; they are contract
checks, not an external driver leak detector.

## RTX 3070 gate — local gate complete, CI pending

`scripts/run-phase4b-rtx3070-acceptance.ps1` runs the deterministic FP16 Llama-2-like
decoder at 16, 23, and 31 layers; CLOCK and LRU at 31 layers; prefetch distances zero
and two; and a two-layer sequence-128 attention smoke. Its fixed geometry is batch 1,
sequence 32, hidden 4096, intermediate 11008, 32 heads, RoPE, two RMSNorms per block,
bias-free attention/SwiGLU projections, untied embedding/head, and no KV cache.

The 16/23/31-layer parameter sizes are 7,000,563,712, 9,833,930,752, and
13,072,064,512 bytes. Oversubscribed runs must show real eviction and frame reuse,
zero weight D2H, bounded scratch, map/SetAccess equality, zero unsafe transitions or
remaps, equal event record/retire totals, zero final live views, reference equality,
complete cleanup, empty diagnostics, and no residual worker. Timing and prefetch effect
are recorded without a speedup threshold.

All six local reports exit `0` and pass the v1 schema, JSONL trace, and semantic
checkers. The three 31-layer result digests agree across CLOCK/LRU and prefetch
distances zero/two. The distance-zero run reports zero submitted and retired prefetches;
the distance-two CLOCK/LRU runs report non-zero, fully retired prefetch work. Every
oversubscribed report shows eviction and frame reuse, all applicable proof flags are
true, cleanup is complete, and diagnostics are empty.

This completes implementation and the local hardware gate. Final delivery still waits
for the full GitHub Actions run; no successful Actions result is claimed here.
