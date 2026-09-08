# Phase 6b.0e — simultaneous launch census

## Result: the public-entry interception gap is confirmed, not fixed

The opt-in census observes CUPTI Runtime/Driver callbacks and GPU kernel activities
in the **same process/run** as the active Driver probe. It keeps the official
llama.cpp executable/backend and all production memory code unchanged. Both final
14B/microbatch-128 runs used eight GPU layers and ordinary CPU offload.

| Observation | Routing | Metadata |
| --- | ---: | ---: |
| GPU kernels observed | 8,776 | 8,776 |
| Kernels correlated with a marked Driver boundary | 96 | 96 |
| Kernels correlated with unmarked launch calls | 8,680 | 8,680 |
| Kernels without a completed launch correlation | 0 | 0 |
| Completed API callback pairs | 95,289 | 95,673 |
| Open API calls / outstanding buffers in footer | 0 / 0 | 0 / 0 |
| Tool errors / reported dropped records | 0 / 0 | 0 / 0 |

There are 39 observed kernel-name groups: two marked CUTLASS layouts and 37 unmarked
groups. The unmarked groups have both `runtime:cudaLaunchKernel` and
`driver:cuLaunchKernel` callback origins. The 384 extra API pairs in metadata mode
are consistent with four diagnostic queries per marked call; those queries are
explicitly outside the native-forward marker.

This replaces the earlier comparison of **separate** captures with simultaneous
evidence. It does not locate an alternative interceptable entry point. A CUPTI
callback's function name is an API label, not proof that execution traversed our
patched export or the public resolver result. In particular, these data do not
justify guessing a private export-table layout, patching an inferred code address,
or treating a static backend source candidate as a compiled memory contract.

The result remains **NO-GO for residency interception**. The unmarked Runtime path,
live function-to-cubin identity, typed tensor/alias/indirect bounds, device ordering,
and live working-set admission remain unresolved. Unchanged 14B/32B generation
through xVRAM and the separate speed gate are still unfinished.

## Correlation and safety contract

Only the real native forward receives a thread-local, monotonically assigned probe
call ID. Metadata queries and log admission do not. Nested public-entry delegation
keeps the outer marker. Callbacks read this integer and CUPTI-owned API metadata;
they do not call CUDA, inspect kernel argument arrays, change arguments or perform
residency operations. GPU activities are linked through CUPTI correlation IDs.

CUPTI can reuse one correlation ID for several Driver calls inside a Runtime call.
Therefore the collector assigns distinct process-local API IDs and parent IDs using
a bounded per-thread stack. The offline analyzer checks enter/exit identity,
parent retirement, depth and marker consistency. It excludes ancestor launch
wrappers when classifying the innermost observed launch boundaries. Conflicting
marked/unmarked siblings or a failed leaf launch with GPU activity are rejected;
they are not silently assigned to the marked call. Buffered activities may arrive
before or after their callbacks.

No CUDA VA, argument values, native handles, PID or OS thread IDs are serialized.
Limits include a 128 MiB trace, eight 1 MiB activity buffers, stack depth 64,
250,000 API instances, 100,000 observed kernels and 4,096 kernel/origin groups.
The subscriber enables Runtime, Driver and concurrent-kernel activities, plus
fatal-error notification. A separate tool thread periodically flushes CUPTI.

Process exit writes only a nonblocking observational footer. It performs no CUDA
synchronization, CUPTI calls or thread joins under process teardown. Consequently
`terminal_complete` is **always false**, even when all observed pairs reconcile and
the footer reports no outstanding buffers. Complete trace/GPU-tail capture and
in-process unload/cleanup safety are not claimed. Missing/lossy/malformed traces,
worker failures and timeouts cannot return diagnostic success. Existing controller
Job Object/process-group timeout and owned-child reap behavior are reused.

New strict report and trace contracts are `xvram.cuda_launch_census` v1. Exit zero
means the selected **partial observation** completed, never a positive execution
decision. All completeness, memory-contract, cubin-binding and oversubscription
proof fields remain false. The companion probe report remains its frozen v1
format; CUPTI provenance belongs only to the new census contract.

## Dependencies and reproduction

The standalone census is disabled by default and has no install rule or production
runtime dependency. It uses the existing pinned Detours package and official
CUPTI 13.3.75 from the
[CUDA 13.3.1 redistributable manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json).
The Windows archive SHA-256 is
`298eb0dfbd7eb63b63aab0533b4ba1ab2542c2ea03842a744973c7f38d0982b3`;
the actually loaded `cupti64_2026.2.1.dll` must match
`9b10d2fafaff1a4dc9e447c4a1355fccb04ee024fa7e7d28c9e4c5ab53347faf`.
The native initializer verifies the loaded module's bytes, not merely a PATH entry.
The same DLL hash was observed in the existing local Toolkit. No system component
was installed, replaced or reconfigured. The fetched package/license stays in the
external dependency directory; only the diagnostic DLL and required CUPTI DLL are
staged together in its build output.

From a Visual Studio x64 developer environment:

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
cmake -S src/compat_launch_probe -B build/compat-launch-census-redist -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DXVRAM_PROBE_CENSUS=ON -DXVRAM_PROBE_FETCH_CUPTI=ON `
  -DFETCHCONTENT_BASE_DIR=D:/xVRAM-dependencies/phase6b0e/cmake-deps `
  '-DXVRAM_PROBE_CUDA_INCLUDE=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include'
cmake --build build/compat-launch-census-redist
python -m xvram.compat_launch_probe run `
  --binary-dir D:/xVRAM-dependencies/phase6b0/llama-b10819 `
  --model-dir D:/xVRAM-dependencies/phase6b0/qwen14b `
  --driver-probe build/compat-launch-census-redist/xvram_driver_launch_census.dll `
  --census-cupti-dir build/compat-launch-census-redist --mode routing `
  --output-dir artifacts/new-census-routing --timeout-seconds 180
python scripts/validate-launch-census.py `
  --report artifacts/new-census-routing/census-report.json `
  --probe-report artifacts/new-census-routing/probe.json `
  --probe-trace artifacts/new-census-routing/launches.jsonl `
  --census-trace artifacts/new-census-routing/census.jsonl
```

Use a different fresh directory for `--mode metadata`. No previous evidence is
overwritten. `python -m xvram.compat_launch_census --help` exposes offline analysis.
Source-free installed Python CLI smoke, CPU stack/TLS tests, malformed/fault report
tests and strict contracts run without a GPU. Windows CI additionally builds the
census using hash-pinned redistributable headers/libraries and checks rejection of
an unpinned host. Linux retains CPU analysis and sanitizer coverage.

Official references: [CUPTI Activity API and correlation](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html),
[callback usage restrictions](https://docs.nvidia.com/cupti/main/main.html), and
[Driver entry-point access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html).

## Acceptance provenance

The [acceptance record](acceptance/phase6b0e-census-20260908.json) pins source,
dependency, built DLL and all final report/trace hashes. Accepted directories are
`accepted-routing-v2` and `accepted-metadata` under
`artifacts/phase6b0e-census-20260908`. Earlier `pilot-routing`, `nested-routing` and
`accepted-routing` directories are **development evidence, not accepted results**:
they predate API nesting or the final separation of CUPTI provenance from the frozen
probe contract. They were retained, not rewritten or silently promoted.

Both final native processes exited zero and their owned process trees were reaped.
Visible output digests match the earlier baseline; this is not tensor-level or
numerical reference equality. No performance advantage is claimed. Version remains
`0.1.0-dev`, and prior Phase 1–6a ABI/schema/export contracts remain unchanged.

Local validation passed 86/86 serial Release CTest, 82/82 serial Debug CTest and
311 audit Python tests (four platform skips), including 16 new census tests.
Both full traces (399,480 records together) passed strict shape/semantic/input
validation. The first parallel Release run encountered a Windows temporary-cwd
deletion lock in an existing audit output-limit test. Complete serial reruns passed
without changing that test, production behavior or deadlines.
