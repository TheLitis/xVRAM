# Phase 6b.0f — legacy-resolved Runtime launch routing

## Result: the observed routing gap is closed for the tested 14B profile

The opt-in diagnostic now marks every **observed** GPU kernel in three new runs of
the unchanged official llama.cpp backend on RTX 3070/WDDM. This is a routing and
live metadata result, **not GO for residency interception or oversubscription**.
All runs still use ordinary CPU offload with eight GPU layers.

| Accepted case | Observed kernels | Marked | Unmarked / uncorrelated | Kernel-name groups | Metadata layouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routing, microbatch 128 | 8,776 | 8,776 | 0 / 0 | 39 | Not queried |
| Metadata, microbatch 128 | 8,776 | 8,776 | 0 / 0 | 39 | 39 |
| Routing, microbatch 1 | 26,860 | 26,860 | 0 / 0 | 15 | Not queried |

Every native process and diagnostic exited zero. Observed API pairs reconcile,
reported dropped records and collector errors are zero, and controllers reaped
their owned process trees. The metadata case queried all 8,776 calls, recording
142,740 parameter entries. Its trace contains 589,572 CUPTI records; the three
census traces total 1,341,476 records. The same visible output digest across these
runs is not tensor-level reference equality or evidence of a speed advantage.

The [previous census](cuda-launch-census.md) remains valid historical evidence:
its two boundaries marked only 96 of 8,776 observed kernels. This follow-up adds
a third, distinct documented resolver target; it does not reinterpret those traces.

## Entry point and handle contract

On the pinned driver, `cuGetProcAddress_v2("cuLaunchKernel", ..., 7000,
CU_GET_PROC_ADDRESS_LEGACY_STREAM, ...)` returns an entry distinct from both the
exported legacy entry and the previously tested per-thread-default-stream entry.
The diagnostic checks target distinctness before committing the hook transaction.
No private export-table layout or guessed implementation address is used.

The third route uses a **declared, pinned-profile contextless-kernel contract**.
CUDA documents that `cuLaunchKernel` can accept a `CUkernel` cast to `CUfunction`.
For metadata, the probe checks the current context and stream context, then calls
`cuKernelGetFunction` and queries the resulting function. It never changes the
current context or substitutes that function in the native launch: the original
handle and all original arguments are forwarded unchanged exactly once.

This is not handle-type inference or an invalid-handle retry strategy. A failed
query, mismatched context or missing function fails closed. Routing-only mode
records the declared handle kind without claiming to have queried it. The metadata
run confirms successful resolution for each observed third-route invocation.
Metadata snapshots are taken per call, not cached by a reusable native handle.

Official contracts: [CUDA execution control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html),
[library/kernel management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html),
and [Driver entry-point access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html).

## Bounded observation, not memory ownership

The same-process CUPTI collector continues to mark only native forwards. Its
callbacks collect metadata and local correlation IDs; they do not query CUDA,
inspect argument arrays, perform residency operations or change arguments.
Nested API instances, ambiguous correlations, missing records and failed launches
retain the existing fail-closed checks. Native addresses and handles are absent
from serialized evidence.

New probe trace v2 describes the three routes and declared handle kind. New census
trace v2 raises the bounded trace capacity to 256 MiB, 1.5 million records and
500,000 API instances. The original v1 trace limits and schemas remain unchanged;
mixed versions are rejected. Both report contracts remain v1. The larger bounded
capacity is needed for per-call metadata queries, not an exemption from loss checks.

The observational teardown footer still sets `terminal_complete: false`.
It does not synchronize CUDA or call CUPTI during process teardown. Therefore
`all_launches_covered` remains false even when **all observed** GPU activities are
marked. Complete GPU-tail capture and safe in-process unload are not established.

Still required before production memory interception:

- Bind each live launch to its compiled module/cubin, not merely a kernel name.
- Prove typed tensor ranges, aliases, indirect accesses, padding and native scratch
  use; a parameter offset/size does not reveal opaque structure members.
- Establish device ordering and chunk-rounded live working-set/budget admission.
- Close terminal capture and extend the active evidence matrix to 32B and all
  required profiles. This follow-up did not run a 32B active-interception gate.
- Only then implement and verify unchanged 14B → 32B execution through xVRAM
  without CPU offload of the main computation, followed by the separate speed gate.

Production residency, original backend bytes, driver, TDR and registry are unchanged.
All Phase 1–6a contracts and previous audit schemas are preserved. Version remains
`0.1.0-dev`.

## Build and reproduce

The standalone option is default OFF and requires the diagnostic census. It uses
the same hash-pinned app-local CUPTI and Detours dependencies documented in the
[census guide](cuda-launch-census.md#dependencies-and-reproduction); it adds no
production dependency or system installation. From a Visual Studio x64 shell:

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
cmake -S src/compat_launch_probe -B build/compat-launch-census-legacy -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DXVRAM_PROBE_CENSUS=ON `
  -DXVRAM_PROBE_LEGACY_RESOLVER=ON -DXVRAM_PROBE_FETCH_CUPTI=ON `
  -DFETCHCONTENT_BASE_DIR=D:/xVRAM-dependencies/phase6b0e/cmake-deps `
  '-DXVRAM_PROBE_CUDA_INCLUDE=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include'
cmake --build build/compat-launch-census-legacy
python -m xvram.compat_launch_probe run `
  --binary-dir D:/xVRAM-dependencies/phase6b0/llama-b10819 `
  --model-dir D:/xVRAM-dependencies/phase6b0/qwen14b `
  --driver-probe build/compat-launch-census-legacy/xvram_driver_launch_census.dll `
  --census-cupti-dir build/compat-launch-census-legacy --mode metadata `
  --output-dir artifacts/new-legacy-metadata --timeout-seconds 180
python scripts/validate-launch-census.py `
  --report artifacts/new-legacy-metadata/census-report.json `
  --probe-report artifacts/new-legacy-metadata/probe.json `
  --probe-trace artifacts/new-legacy-metadata/launches.jsonl `
  --census-trace artifacts/new-legacy-metadata/census.jsonl
```

Use fresh output directories for `--mode routing` and `--microbatch 1`. The combined
validator checks both reports, every record of both version-selected traces, their
semantics and exact input recomputation. Windows CI builds old/new diagnostic
profiles and rejects an unpinned host; Linux retains no-driver and sanitizer tests.

## Acceptance provenance

The [acceptance record](acceptance/phase6b0f-legacy-20260908.json) pins source commit
`e019210ff6fee4e1ed6bb895c6796bc8d669cbbb`, the built diagnostic DLL and each final
report/trace. Only its three `accepted-*` directories are final results.
Earlier trials exposed a contextless-handle metadata error and the old 128 MiB
trace cap; those failures were retained, not relabeled as success. The successful
development `metadata` directory predates the final context-resolution refactor
and is not substituted for final acceptance.

Local validation passed 86/86 serial Release CTest, 82/82 serial Debug CTest and
314 audit Python tests (four platform skips), including 13 probe and 18 census
tests. All three complete report/trace pairs passed strict validation; installed
Python CLI smoke and frozen-contract checks passed. No residual llama process was
observed after the final runs.
