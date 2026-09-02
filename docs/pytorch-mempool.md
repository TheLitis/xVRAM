# PyTorch MemPool integration

Phase 4a adds an experimental PyTorch adapter around
`torch.cuda.memory.CUDAPluggableAllocator` and `torch.cuda.MemPool`. The adapter lets
selected PyTorch allocations use CUDA Driver API virtual-memory segments while keeping
PyTorch's existing caching allocator, tensor ownership, and stream tracking in control.

This is a correctness-first integration boundary. Every native segment is fully
resident in VRAM for its entire lifetime. It does **not** yet connect PyTorch tensors to
the pageable-RAM residency cache from Phase 2, and therefore does not increase the
amount of tensor data that can be resident at once.

## Requirements and support status

The native plugin requires:

- a 64-bit Windows or Linux build;
- CUDA 13.0 or newer headers when building xVRAM and a compatible NVIDIA driver at
  runtime;
- a CUDA device with Driver API VMM support;
- a CUDA-enabled PyTorch build exposing `CUDAPluggableAllocator`, `MemPool`, and
  `use_mem_pool`.

PyTorch's public documentation currently describes `CUDAPluggableAllocator` as
Unix-only. Linux is consequently the upstream-supported integration path. xVRAM also
implements a Windows DLL path and tests the four-argument allocation/free callbacks,
but that path is experimental and is not an upstream PyTorch portability guarantee.
The Python adapter uses an absolute DLL path and keeps both the plugin directory and
PyTorch's `torch/lib` DLL search-directory handles alive with the allocator.

The relevant upstream interfaces are
[`CUDAPluggableAllocator`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.CUDAPluggableAllocator.html)
and the [`MemPool` CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Build and install

With a local CUDA Toolkit, configure a test build without downloading headers:

```powershell
cmake -S . -B build\phase4 `
  -DXVRAM_BUILD_TESTS=ON `
  -DXVRAM_FETCH_CUDA_HEADERS=OFF
cmake --build build\phase4 --config Release --target xvram_torch_allocator
```

If CUDA headers are not installed, omit `-DXVRAM_FETCH_CUDA_HEADERS=OFF`; CMake uses the
project's hash-pinned official NVIDIA header redistributable. The resulting native
library is normally:

- `build\phase4\Release\xvram_torch_allocator.dll` with a multi-configuration Windows
  generator;
- `build/phase4/libxvram_torch_allocator.so` with a single-configuration Linux
  generator.

The CMake install step installs the shared library, C telemetry header, and CMake target
`xVRAM::torch_allocator`:

```powershell
cmake --install build\phase4 --config Release --prefix build\install
```

The CMake step installs only native artifacts. Install the Python package independently
from the source tree, or add its directory to `PYTHONPATH` during development:

```powershell
python -m pip install .\python
```

## Python usage

`create_mem_pool()` loads the native plugin, creates a
`CUDAPluggableAllocator`, and retains both objects for the lifetime of the returned
pool. The native path must be absolute after resolution. It can be supplied directly or
through `XVRAM_TORCH_ALLOCATOR_LIBRARY`.

```powershell
$env:PYTHONPATH = (Resolve-Path .\python).Path
$env:XVRAM_TORCH_ALLOCATOR_LIBRARY = `
  (Resolve-Path .\build\phase4\Release\xvram_torch_allocator.dll).Path
```

```python
import torch
from xvram import create_mem_pool

pool = create_mem_pool()
pool.reset_stats()

with pool:
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    y = x.square()
    result = y.sum().cpu()

torch.cuda.synchronize()
stats = pool.get_stats()
assert stats.allocation_failures == 0
assert stats.maps == stats.set_access_calls
assert stats.unsafe_unmaps == 0
assert stats.quarantined_segments == 0
print(result.item(), stats.to_dict())
```

`use_mem_pool` is a scoped, thread-local routing decision. Only allocations made inside
the context use that pool; entering the context does not migrate existing tensors.
Tensors may outlive the context, so leaving `with pool:` is not the same as destroying
the pool or releasing all of its cached segments.

For applications that need the underlying objects separately:

```python
from xvram import load

allocator = load("/absolute/path/to/libxvram_torch_allocator.so")
pool = allocator.create_mem_pool()
with pool.use():
    # New CUDA allocations are routed through this pool on the current thread.
    ...
```

## Native segment safety boundary

PyTorch calls the exported `xvram_torch_alloc(size, device, stream)` and
`xvram_torch_free(pointer, size, device, stream)` functions. A successful allocation:

1. validates the current PyTorch CUDA context and device;
2. rounds the requested size up to the device's minimum VMM granularity;
3. reserves one stable CUDA virtual-address range;
4. creates one same-sized physical device allocation handle;
5. maps the complete handle and applies read/write access with `cuMemSetAccess`;
6. returns the stable address to PyTorch.

There is no sparse mapping in this version: mapped bytes and physical bytes are equal.
There is also no pageable backing, eviction, prefetch, write-back, or physical-handle
reuse across live PyTorch segments.

Before a segment is unmapped, the free callback records and synchronizes a CUDA event
on its recorded allocation stream. It then unmaps the full mapped range, releases the
physical handle, and frees the exact original VA reservation. If an event, unmap, or
context-activation boundary cannot be proven safe, the segment is quarantined rather
than unsafely reused. Failure to restore the caller's context after a segment is already
fully released remains a reported cleanup failure. Allocation callbacks return null on
failure, and C++ exceptions never cross the PyTorch callback ABI.

New segment allocation or release during CUDA stream capture is rejected. An operation
that is already satisfied from PyTorch's existing pool may not invoke the plugin, but
xVRAM makes no CUDA Graph capture guarantee: a capture miss or any callback requiring a
new/released VMM segment is unsupported.

## Stream ownership and `record_stream`

The native callback can fence the stream PyTorch passes for its backing segment, but it
cannot discover arbitrary uses performed later by another stream or by a custom CUDA
extension. The normal PyTorch caching-allocator contract therefore remains mandatory:
record every non-owning stream that uses a tensor's storage.

```python
with pool:
    x = torch.empty(1_000_000, device="cuda")
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        y = x.square()
    x.record_stream(consumer)

consumer.synchronize()
```

Call `record_stream()` before the tensor can be destroyed or its storage returned to
the pool. Views share their base storage and need the same lifetime discipline. Code
that exports raw pointers, launches work outside PyTorch's stream tracking, or uses a
third-party extension must establish an equivalent event boundary itself; lifetime and
FX hints do not substitute for that boundary.

The stream handle recorded for a newly allocated segment must remain valid until
PyTorch releases that segment. Allocating through an `ExternalStream` or a custom
runtime and then destroying the underlying CUDA stream while the pool still caches its
segment is unsupported; xVRAM will quarantine the segment if the final event fence can
no longer be recorded.

## Pool lifetime and destruction

The pool and its allocator owner must outlive every tensor or storage allocated from
the pool. `XvramMemPool` holds its `XvramTorchAllocator`, so keeping the pool alive is
sufficient; do not manually unload the native library.

Before releasing the last pool reference:

1. stop submitting work that can touch pool-backed tensors;
2. synchronize every CUDA stream that used those tensors;
3. drop tensors, views, and external storage references;
4. let PyTorch destroy the raw `MemPool`, then release the allocator owner.

PyTorch may keep native segments cached after tensors are deleted, so
`active_segments` can remain nonzero until the raw pool itself is destroyed. Exiting
the context only restores the previous allocation route. If telemetry reports a
quarantined segment, treat the process as the cleanup boundary rather than unloading
and reloading the plugin in place.

## Telemetry

The size-tagged C ABI in `include/xvram/torch_allocator.h` exports read-only v1
telemetry through `xvram_torch_get_stats`, reset through `xvram_torch_reset_stats`, and
last-error detail through `xvram_torch_get_last_error`. The Python adapter exposes the
counter snapshot as `pool.get_stats()` or `allocator.get_stats()`.

Important groups are:

- allocation/free calls, failures, requested and granularity-rounded mapped bytes;
- current/peak segment counts and bytes;
- reservations, handles, maps, `SetAccess` calls, event boundaries, unmaps, releases,
  and reservation frees;
- capture, context, size, and stream mismatches;
- OOM, quarantine, unsafe-unmap, and last native status/error fields.

For a healthy drained run, allocation failures, free failures, quarantines, and unsafe
unmaps are zero. Every map has a matching `SetAccess`. While segments are still live,
map/unmap and create/release totals need not yet balance.

`reset_stats()` resets cumulative counters but preserves gauges for segments that are
still live; it is intended to establish a workload-local baseline, not to release
memory.

## Lifetime classification and FX plans

Phase 4a includes metadata-only planning helpers. They do not inspect CUDA addresses,
move tensors, or authorize eviction.

```python
from xvram import TensorRole, classify_module_tensors, classify_tensor

input_hint = classify_tensor(role=TensorRole.INPUT)       # external
activation_hint = classify_tensor(role="activation")     # iteration
persistent = classify_module_tensors(model)               # parameters and buffers
```

Classification is deliberately explicit. `requires_grad` and `is_leaf` cannot reliably
distinguish parameters from caller-owned inputs, so an unlabelled tensor remains
`unknown`. Module parameters and registered buffers are persistent. Optimizer state is
only classified when the caller explicitly supplies `role="optimizer_state"`; the
helper does not walk optimizer objects.

FX analysis produces deterministic next-use, last-use, release-candidate, and bounded
prefetch-candidate metadata:

```python
import torch.fx
from xvram import analyze_next_uses

graph_module = torch.fx.symbolic_trace(model)
plan = analyze_next_uses(graph_module, prefetch_distance=2)
for step in plan.steps:
    print(step.node_name, step.release_candidates, step.prefetch_candidates)
```

Placeholders, `get_attr` values, and graph outputs are never release candidates. The
step metadata is interpreted after the named node submits its work; reclamation still
waits for that node's completion event. Dead intermediate results are release candidates
on their own step, while prefetch candidates must have been produced earlier. The plan
reflects only the static FX graph; aliases, mutation, dynamic control flow,
autograd, and completion of asynchronously launched CUDA work require separate runtime
proof. Consequently `release_candidates` and `prefetch_candidates` are advisory facts,
not executable residency commands. Integrating them with inference dispatch and
event-safe residency scheduling is later Phase 4 work.

## Current limitations

Phase 4a intentionally does not provide:

- PyTorch oversubscription, pageable tensor backing, eviction, write-back, or handle
  reuse;
- transparent execution of models whose live tensor working set exceeds VRAM;
- allocation misses or segment release during CUDA Graph capture;
- CUDA IPC/shared-storage export, NCCL integration, distributed execution, or
  multi-GPU allocation/migration support;
- interception of raw-pointer use by custom CUDA extensions;
- automatic optimizer-state, backward-graph, alias, or mutation analysis;
- an inference executor that acts on FX next-use hints.

The Phase 2 residency cache and the Phase 4a MemPool allocator have compatible safety
goals but are separate runtimes. Joining them safely requires graph-aware pinning and a
completed event for every PyTorch stream user before a physical frame can be unmapped
or reused.

## Acceptance checks

CPU-only allocator, adapter, lifetime, and FX tests run through CTest:

```powershell
cmake --build build\phase4 --config Release
ctest --test-dir build\phase4 -C Release --output-on-failure -R "xvram[.]torch"
```

On a CUDA host with a compatible PyTorch build, run the focused callback smoke directly:

```powershell
$env:PYTHONPATH = (Resolve-Path .\python).Path
$env:XVRAM_TORCH_ALLOCATOR_LIBRARY = `
  (Resolve-Path .\build\phase4\Release\xvram_torch_allocator.dll).Path
python .\tests\python\test_torch_allocator_integration.py -v
```

Linux uses the equivalent environment variables:

```bash
export PYTHONPATH="$PWD/python"
export XVRAM_TORCH_ALLOCATOR_LIBRARY="$PWD/build/phase4/libxvram_torch_allocator.so"
python tests/python/test_torch_allocator_integration.py -v
```

Acceptance requires at least one native allocation callback, zero allocation failures,
`maps == set_access_calls`, zero unsafe unmaps, and zero quarantined segments. A
successful smoke validates the resident-only callback boundary; it is not an
oversubscription acceptance result.
