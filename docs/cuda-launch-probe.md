# Phase 6b.0d — active launch diagnostics, not residency interception

## Approved boundary and result

The user approved active diagnostic interception after the passive audit reached
its structural stop. The experiment keeps the official executable and original
CUDA backend byte-identical, uses ordinary resident CUDA memory, and does not
attach to an existing process. Production residency and Phase 1–6a exports,
ABIs and schemas are unchanged. Version remains `0.1.0-dev`.

There are two separate mechanisms:

1. An optional app-local CUDA Runtime forwarder, staged in a **new** directory.
   It preserves the pinned runtime's 482 named exports/ordinals and observes only
   `cudaLaunchKernel` and `cudaLaunchKernelExC`. Other exports are native PE
   forwarders; private compiler registration exports are never decoded. The
   pinned `ggml-cuda.dll` has no cudart DLL import. The negative-control run
   completed ordinary generation but produced no intercepted launches. This
   mechanism is therefore **not a route to the pinned stock backend**.
2. A Windows diagnostic DLL loaded through CUDA's `InitializeInjection` path in
   the controller-owned pinned `llama-completion.exe`. It instruments two public
   Driver entry points in that process using Microsoft Detours: exported
   `cuLaunchKernel` and the public resolver result for `cuLaunchKernel`, CUDA
   version 7000, per-thread-default-stream flags. It does not parse private
   Driver export tables or replace a backend or kernel.

The Driver pilot observes 96 successful calls. Metadata mode resolves the live
function's native module and collects its name, parameter count and parameter
offsets/sizes on every call. Two CUTLASS function layouts are observed, 48 calls
each, with one by-value parameter per layout. The content or members of these
opaque structures are **not** inspected. There is no inference of tensor bounds.

This is real but **partial** active interception. The earlier separate passive
14B/microbatch-128 pilot observed 8,776 GPU activities. The new active experiment
is not a simultaneous GPU census and must not claim a precise coverage percentage.
Resolving the public per-thread entry point did not increase the 96-call count.
The remaining Runtime-origin routes still require investigation; a callback's API
name does not prove that execution crossed the corresponding exported entry point.

## Safety and evidence semantics

The Driver DLL checks the executable and backend hashes before installing hooks.
It enrolls an at-most-256-thread snapshot of its own process in a Detours transaction;
enrollment/attach failures abort installation. This is a controlled startup
experiment, not a general-purpose late-attach or concurrent-library-unload profile.
No other process is patched. Driver files, TDR, registry and NVIDIA settings are
not changed. Hooks/trampolines live until the owned process exits; this is not an
in-process unload API or a proof of SDK resource cleanup.

No CUDA query is made from a CUPTI callback. The active Driver probe has no CUPTI
dependency. It runs diagnostic queries at the intercepted public function boundary,
on the calling thread, without changing its context or reading kernel argument
arrays. There is no function-handle metadata cache: every invocation receives a
new snapshot, avoiding stale cached layouts after handle reuse. This does not
establish module-lifetime generations across unobserved unload paths.

The common CPU core limits layouts to 256 parameters and a 64 KiB bank, checks
overlap/overflow, owns copied names, and requires metadata/log admission before
exactly one native forward. Query/admission failures stop that call and poison
subsequent observed calls. A trace write failure after forwarding never replays
the operation. Nested public-entry delegation preserves the native forwarding
chain without recursively issuing metadata queries; nested coverage is not claimed.
Diagnostic queries may perturb lazy loading, error state and timing, so this is
not advertised as transparent production error semantics or a performance test.

The trace is bounded at 128 MiB. It contains sequential `setup` (Driver only),
`begin`, `parameter`, and `end` records. `end` means the host API returned, **not**
GPU completion. CUDA addresses, argument values, native function/module/stream
handles and process/thread identifiers are never serialized. Each snapshot
confirms only that the queried native module was non-null; it is not a
native-to-CUPTI module-ID or cubin-SHA binding.

The controller reuses the tested Windows Job Object/Linux process-group primitive;
native execution is Windows-only. It has a configurable 1–900 second total run
deadline, not an invented retirement heartbeat. The owned worker/tree is reaped
on timeout. Missing, truncated, inconsistent or empty traces cannot return success.
Strict new report/trace schemas use `xvram.cuda_launch_probe` v1. All readiness,
full-coverage, cubin-binding, semantic-range, GPU-completion and oversubscription
proof flags remain false. Exit zero means the **selected diagnostic completed**.

## Reproduce the Driver pilot

Run from a Visual Studio x64 developer environment with installed CUDA headers.
Dependencies and experimental binaries are not installed into the system.
The standalone target is excluded from normal SDK installation.

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
$env:CUDA_PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3'
cmake -S src/compat_launch_probe -B build/compat-launch-probe -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DFETCHCONTENT_BASE_DIR=D:/xVRAM-dependencies/phase6b0d/cmake-deps
cmake --build build/compat-launch-probe
python -m xvram.compat_launch_probe run `
  --binary-dir D:/xVRAM-dependencies/phase6b0/llama-b10819 `
  --model-dir D:/xVRAM-dependencies/phase6b0/qwen14b `
  --driver-probe build/compat-launch-probe/xvram_driver_launch_probe.dll `
  --mode metadata --output-dir artifacts/new-driver-metadata --timeout-seconds 180
python scripts/validate-launch-probe.py `
  --report artifacts/new-driver-metadata/probe.json `
  --trace artifacts/new-driver-metadata/launches.jsonl
```

Use `--mode routing` in a different fresh directory for the no-query comparison.
The profile remains 14B, eight GPU-offload layers, microbatch 128 (or explicit 1),
context 2048, FP16 KV, 32 generated-token limit, graphs/FA/speculation/context
shifting disabled. These are **ordinary CPU-offload diagnostic runs**, never
xVRAM oversubscription. Equal visible-output digests do not establish tensor-level
or numerical equality. No speed benefit is claimed.

For the negative-control Runtime proxy, configure
`XVRAM_PROBE_BUILD_RUNTIME_PROXY=ON` and `XVRAM_PROBE_NATIVE_CUDART` to the pinned
runtime. `python -m xvram.compat_launch_probe stage --help` describes the fresh
directory staging command. Do not place the cudart-named proxy in the original
dependency directory. The native runtime is copied, not altered, under a distinct
name; an import-library definition uses that same name to avoid a proxy self-import.

## Dependencies and next gate

Microsoft Detours v4.0.1 is pinned to commit
`e4bfd6b03e50de46b47abfbd1e46b384f0c5f833` and archive SHA-256
`fef0b17c5c3c9356fb8ba7366a3b196149daadd131fb6a92abeda1cb4d3b9b85`.
Its MIT license is copied alongside the build. Third-party Detours compilation is
separate from the warning-as-error xVRAM targets. CI builds the active DLL on
Windows and verifies rejection of an unpinned host; CPU core, parser, schema,
fault-report and timeout/reap tests also run without a GPU.

The next gate is actual routing coverage of the missing Runtime-origin launches,
then live function-to-binary identity and typed memory/ordering contracts. This
prototype does not authorize promoting names/layouts into source semantics. The
production memory interceptor must remain disabled until these obligations and
working-set admission are proven. 32B execution through xVRAM and the speed gate
remain unfinished.

Primary API references: [CUDA execution queries](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html),
[Driver entry-point resolution](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html),
[CUPTI callback restrictions and injection sample](https://docs.nvidia.com/cupti/main/main.html),
[Microsoft Detours transactions](https://github.com/microsoft/Detours/wiki/Using-Detours),
and [PE export forwarders](https://learn.microsoft.com/en-us/cpp/build/reference/exports?view=msvc-170).

## Recorded validation — 2026-09-08

The [acceptance record](acceptance/phase6b0d-launch-probe-20260908.json) pins the
source snapshot, built DLL, dependency, reports and traces. Both accepted runs
returned zero, had zero native API errors, passed strict schema/semantic checks,
and reaped/drained their owned process trees. Each observed 96 host API pairs;
metadata mode recorded 96 parameter snapshots. Their visible-output digests agree.

Local validation passed 86/86 Release CTest, 82/82 serial Debug CTest and 295 audit
Python tests (four platform skips). The standalone DLL built warning-clean and
refused initialization in an unpinned Python host. The normal build retains all
prior frozen-contract tests. Development negative-control and intermediate runs
are kept separately from `accepted-routing` / `accepted-metadata` and are not
promoted to the final strict report contract.

CI build wiring was also checked locally against the split CUDA runtime/CRT
header packages: the Runtime proxy compile-check includes both roots. The
Stable-ABI job explicitly builds both audit test executables before running the
five-test no-driver audit subset (5/5 passed locally). These build-only fixes do
not change the source/DLL hashes of the recorded hardware snapshot.
