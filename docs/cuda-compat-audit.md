# Phase 6b.0: transparent-compatibility audit

The long-term xVRAM goal is unchanged-application execution with RAM as the backing
tier and VRAM as a bounded, event-safe cache. Logical capacity has no fixed `2x` or
`4x` product ceiling. CUDA virtual-address space, live host/device budgets, and the
working set of an operation still impose real limits.

Phase 6b.0 is the diagnostic prerequisite for that execution path. It does **not**
implement a CUDA interceptor or claim that the pinned application already runs through
xVRAM. It leaves production residency, the explicit Phase 6a adapter, and all earlier
ABI/schema/export contracts unchanged. A useful completed audit may conclude `NO-GO`.

## Pinned application profile

The first practical target is an unchanged official llama.cpp executable **and its
original CUDA backend**. A rebuilt host application or a new GGML backend is not an
equivalent acceptance result.

| Component | Pinned identity |
| --- | --- |
| llama.cpp | [`b10819`](https://github.com/ggml-org/llama.cpp/releases/tag/b10819), commit `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`, Windows x64 CUDA 13.3 package |
| Qwen2.5-14B-Instruct GGUF | Revision `b466e1f8c07172155743e8e1307507d8a4f91fbd`, Q4_K_M shards only |
| Qwen2.5-32B-Instruct GGUF | Revision `a15e3cc10f8bbb2c0af6f8f1f34a32e3b060c09d`, Q4_K_M shards only |
| Hardware gate | Windows WDDM, one RTX 3070 |

Model sources are the publisher's [14B repository](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF)
and [32B repository](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF).
Release identity is not enough by itself: the evidence records the SHA-256 of each
executable, backend, selected NVIDIA library, and model shard. Model shards must match
the hashes published for the pinned revision. Downloads and tools remain outside
tracked source; no system CUDA, driver, TDR, registry, or NVIDIA setting is changed.

The bounded profile uses one CLI request, one sequence, FP16 KV, and at most 2048
context tokens. CUDA Graphs, Flash Attention, speculative decoding, and context
shifting are disabled. Microbatch 1 and 128 are separate cases. Both real prefill and
repeated decode, including KV-cache lifetimes, must be observed. xVRAM compression
and additional quantization are not part of this profile.

The immutable dependency manifest is packaged as
[`compat_audit_profile.json`](../python/xvram/compat_audit_profile.json). It pins both
archive hashes, all 54 extracted executable/DLL hashes, and the three 14B/five 32B
shards with their exact byte counts and publisher hashes. Prepare the files with an
already installed `hf` CLI; the script does not install tools:

```powershell
./scripts/fetch-phase6b-audit-dependencies.ps1 `
  -DependencyRoot D:/xVRAM-dependencies/phase6b0 -HfPath C:/path/to/hf.exe -Models both
./scripts/fetch-phase6b-audit-dependencies.ps1 `
  -DependencyRoot D:/xVRAM-dependencies/phase6b0 -Models both -VerifyOnly
```

The downloader verifies existing files before any download, refuses to overwrite
incorrect bytes, verifies archives before extraction, and requests each model shard at
the exact revision. It does not merge shards, run the application, or alter system
CUDA. `-Models 14b`, `32b`, or `none` selects a smaller preparation/verification set;
both models remain mandatory for the complete hardware audit.

## Tools and evidence flow

The Python entry point is installed with the existing package:

```text
python -m pip install ./python
xvram-compat-audit --help
```

The same commands are available as `python -m xvram.compat_audit`. Inventory and
analysis are offline, no-driver operations; importing the tool does not load the SDK,
CUDA, or PyTorch. Capture is explicitly requested and supervises only its child
application. A timeout, crash, or incomplete collector close cannot become a positive
compatibility verdict. Baseline application output and profiling artifacts are evidence,
not a replacement for normalized contract validation.

Select `--stage inventory`, `capture`, `analyze`, or `all`; inventory is the default.
`--binary-dir` points to the extracted official package, and `--model-dir` to the
verified Q4_K_M shards for `--model 14b` or `32b`. `--output-dir` separates diagnostic
artifacts from source. Capture accepts `--capture-mode baseline|cupti|nsys`, an explicit
`--collector` or `--nsys` path, and a 900-second default deadline. Analysis accepts
repeatable `--input-trace` paths. `--json <path|->`, `--compact-json`, and `--no-text`
control the report output independently.

`--application completion` is the default automated frontend: the unchanged official
`llama-completion.exe` provides deterministic batch operation without the chat UI.
`--application cli` selects the separately pinned official `llama-cli.exe`; both use
the same unchanged CUDA backend. For a validated, successful **unprofiled completion**
run with visible generated output, `ttft_ms` is the observed time from process start
to the first visible generated output, as identified by the pinned frontend's
[generated-piece output path](https://github.com/ggml-org/llama.cpp/blob/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/tools/completion/completion.cpp#L704-L725).
It includes startup/loading/warmup and output transport; it is not GPU token-completion
latency. Other cases retain `first_stdout_ms` separately and leave TTFT unknown:
banners, terminal markers, buffered output, the chat UI, or a profiler can make the
first stdout byte a different event.

For example, the following explicitly requests a CUPTI observation with eight layers
offloaded to GPU; it is a diagnostic CPU-offload baseline, not all-GPU inference:

```powershell
xvram-compat-audit --stage all --model 14b `
  --binary-dir D:/xVRAM-dependencies/phase6b0/llama-b10819 `
  --model-dir D:/xVRAM-dependencies/phase6b0/qwen14b `
  --capture-mode cupti --collector C:/path/to/xvram_compat_audit_collector.dll `
  --gpu-layers 8 --microbatch 128 --context-size 2048 --generate 32 `
  --output-dir artifacts/phase6b0-14b-ub128 --json artifacts/phase6b0-14b-ub128.json
```

The package/shard directories and collector path in the command are placeholders for
the verified local files; the tool checks their identities before launch. Use a separate
output directory per model, microbatch, and capture mode. The defaults are eight GPU
layers, microbatch 128, context 2048, and 32 generated tokens. An optional
`--prompt-file` fixes a local prompt. Repeat with `--model 32b` and microbatch 1, and
collect `--capture-mode baseline` runs separately for performance measurements.

The optional host-C++ CUPTI collector can be built separately or through the root
project. For a root build, supply the installed CUDA/CUPTI development package:

```text
cmake -S . -B build/audit -DXVRAM_BUILD_COMPAT_AUDIT_COLLECTOR=ON
cmake --build build/audit --config Release --target xvram_compat_audit_collector
```

`XVRAM_CUPTI_ROOT` selects a separate CUPTI package root when needed. The option
defaults to `OFF`, so regular SDK consumers do not acquire a profiling dependency.
The collector is a diagnostic library, not a public SDK ABI or a native fallback.

CUPTI and Nsight Systems are used only for observation. Callbacks record data; they
do not call residency methods, submit GPU work, modify arguments, or map/unmap
application memory. See NVIDIA's [CUPTI callback restrictions](https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api).
Profiler availability, collection errors, and dropped records must remain visible.

## What must be established

The audit separates three evidence levels: **observed**, **source-proven**, and
**unresolved**. One cannot be silently promoted into another.

- Inventory actual imports, dynamically obtained API entry points, kernel registration,
  and Runtime/Driver/cuBLAS use for the pinned binaries.
- Correlate allocations, frees, tensor subranges, aliases, host-pinned memory, streams,
  events, and llama.cpp's native VMM scratch pool across the complete run.
- Identify each launch's module/kernel and argument ABI, declared reads/writes,
  padding, indirect accesses, and scratch. An allocation size or a pointer value alone
  is not proof of a tensor's accessible range.
- Compute chunk-rounded working sets after native scratch/workspace reservations.
  Preserve observed context ownership and ordering instead of assuming that the
  isolated synchronous Phase 6a SGEMM worker can run the unchanged backend.
- Account for setup, warmup, prefill, repeated decode, and teardown. Missing lifecycle
  stages or ambiguous lifetime/event correlation leave coverage incomplete.

The current observation boundary does not infer arbitrary kernel semantics. Unknown
or unproven tensor bounds remain explicit blockers even if a run succeeds natively.

## Initial evidence and the remaining readiness gaps

The pinned `ggml-cuda.dll` PE inventory directly imports `cublas64_13.dll` entry
points including `cublasGemmEx`, batched/strided-batched variants, SGEMM, and
stream/workspace controls. It also imports `GetProcAddress` and `LoadLibrary` variants.
No cudart or CUDA Driver import descriptor appears in **that DLL's PE import table**.
This is not evidence of absent runtime/driver use, static linking, or a complete list
of callable CUDA entry points: dynamic resolution remains observable work.

The initial local 14B diagnostic run observed `nvcudart_hybrid64.dll`, `nvcuda.dll`,
`nvcuda64.dll`, and the pinned cuBLAS pair in its loaded-module snapshots. Its CUPTI
trace contains Driver API calls and native VMM operations; this corroborates the need
to follow actual resolution and memory lifetimes instead of treating the import table
as a complete execution contract. The observation is not the full model/microbatch
matrix, and the native CPU-offload run is not an xVRAM oversubscription proof.

Two blockers are explicit at the standard observer boundary:

- Runtime callback parameter bytes are **not** the launched kernel's argument ABI.
  Kernel/module identity and allocation extents do not establish quantized tensor
  subranges, indirect reads, padding, writes, or scratch requirements.
- The initial Windows collector trace has no finalized end marker. Process exit and
  a last visible API callback do not prove all CUPTI activity buffers were flushed or
  every teardown event was observed. The audit must retain incomplete coverage.

These findings mean **NO-GO for implementing this profile from the current evidence**,
not proof that transparent compatibility is impossible. The next investigation is to:

1. Reconcile pinned on-disk imports with observed loaded libraries and actual dynamic
   API resolution. Preserve the distinction between module snapshots, API callbacks,
   and complete routing coverage.
2. Build source-backed argument/access contracts for the kernels actually selected by
   each bounded Qwen case, including quantization layout, aliases, index bounds, and
   scratch. Match those contracts to exact module/kernel identities and reject gaps.
3. Establish complete collection/teardown evidence using supported profiler lifecycles
   and cross-checks. A finalized Nsight export and its aggregate summary can corroborate
   counts/timings but cannot supply missing kernel access ranges or event lifetimes.
4. Reconcile llama.cpp's own VMM pool and observed context/stream relationships with
   chunk-rounded working sets and native workspace reservations. Multiple context IDs
   or streams cannot be collapsed into the current isolated serial Phase 6a worker by
   assumption; that worker's FP32 SGEMM profile is not the unchanged quantized backend.

This investigation does not authorize hooks, kernel replacement, a rebuilt backend,
or a different GGML plugin. Any proposed change to that integration boundary needs a
separate decision. The audit's job is to produce the evidence and precise blockers
for that decision, not to silently substitute an easier application profile.

## Reports, privacy, and interpretation

The new report contract is `xvram.cuda_compat_audit` v1, validated by
[`cuda-compat-audit-v1.schema.json`](../schemas/cuda-compat-audit-v1.schema.json).
The normalized trace uses `xvram.cuda_compat_audit_trace` v1 and
[`cuda-compat-audit-trace-v1.schema.json`](../schemas/cuda-compat-audit-trace-v1.schema.json).
Reports carry provenance, configuration, coverage, memory bounds, unresolved cases,
cleanup evidence, and the audit decision.

Addresses and native handles are converted to process-local IDs before normalized
serialization. These IDs describe relationships, not reusable pointers. Profiling
tools' own native artifacts and raw application logs have separate privacy properties:
keep them local and review them before sharing. Hashes and software/configuration
details are identifying evidence, not anonymization.

In particular, Nsight's `.nsys-rep` and exported SQLite are external raw diagnostic
artifacts, not normalized xVRAM reports. They may contain addresses and identifying
process details. Keep them in the local capture directory; do not publish them as a
privacy-safe trace. Normalization must remove raw pointers/handles before any such
data enters the xVRAM report/trace contract.

Loaded-library provenance uses bounded OS snapshots of the controller-owned child,
not an enumeration of unrelated applications. Windows queries its module paths only
internally to obtain approved basenames and file hashes. These snapshots can miss a
briefly loaded module or race unloading; a failed query is unknown, not verified
absence. The Windows [module enumeration API](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-enumprocessmodulesex)
documents those limitations. Hashes identify on-disk files at observation/verification
time, not a cryptographic measurement of the process's mapped instruction bytes.

`GO` requires established interception points, memory contracts, execution order, and
admissible working sets for **the whole declared profile**, with no unresolved use of
future managed pointers. Empty or truncated traces, dropped records, unknown kernel
footprints, missing teardown, or contradictory evidence exclude `GO`.

`NO-GO` is a compatibility finding, not permission to rebuild the backend, substitute
kernels, enable CPU fallback, or install a speculative interception mechanism. Record
the exact blockers and the required next investigation. Passing JSON Schema validation
only establishes report shape; it does not establish compatibility or oversubscription.

## Hardware observations and baselines

Observe both 14B and 32B at microbatch 1 and 128, from loading through cleanup. An
ordinary application case that does not fit VRAM may use explicitly labelled CPU
offload for baseline research. Such a case is **never** an xVRAM oversubscription proof
and cannot demonstrate all-GPU execution.

Measure unprofiled baselines separately from diagnostic captures: model loading,
time-to-first-token, prefill throughput, decode throughput, and RAM/VRAM consumption.
Profiling overhead must not be presented as application performance. A native success
does not by itself prove that all future managed ranges are resident-safe.

RAM samples describe the direct command process only: with `nsys` wrapping the
application, that process is the profiler, not automatically llama.cpp. GPU samples
from `nvidia-smi` describe the whole device, not this process and not the WDDM admission
budget; device ordering must be reconciled separately. Sampled maxima are observations,
not a guarantee that a short-lived peak was seen. Missing measurements stay unknown.

The source contains tooling, not a standing hardware certificate. The
[roadmap](roadmap.md) records the actual completed evidence and remaining blockers;
only saved, validated observations from the pinned binaries/models count toward the
gate. CI does not download multi-gigabyte models or claim RTX acceptance.

## Tests and delivery boundary

The focused no-driver suite can run without the collector, GPU, or profiler:

```text
python -m unittest discover -s tests/python -p "test_compat_audit*.py"
ctest --test-dir build --build-config Release --output-on-failure -R "^xvram[.]compat-audit[.]"
```

Install the Python package first or set `PYTHONPATH` to `python`. Tests cover malformed
imports/manifests and traces, range/overflow/lifetime validation, event correlation,
privacy, coverage failures, controller termination, and conservative decision semantics.
The existing Phase 1–6a contract checks still run unchanged. Windows/Linux Debug and
Release, Linux Clang sanitizers, and the Stable-ABI regression retain no-driver audit
coverage.

After the audit, Phase 6b.1 requires a separately approved implementation of the proven
profile and real 14B then 32B generation without CPU offload of the main computation.
Only then does a separate performance stage establish an advantage over a tuned native
baseline. Broader models, kernels, server operation, graphs, and other applications are
later compatibility expansions, not implied by this diagnostic tool.

## Recorded local results

The audit concluded **NO-GO** on 2026-09-06 on Windows x64/RTX 3070/WDDM, driver
616.56. The official package and all Q4_K_M shard hashes matched the pinned manifest.
Both microbatches used context 2048, one sequence, FP16 KV, eight GPU layers, the fixed
default prompt (161 tokens), greedy sampling seed 424242, and 32 generated tokens
(31 measured decode iterations). CUDA Graphs/FA/speculation/context shifting were
disabled by the recorded profile. The native backend's `USE_GRAPHS` build capability
and GGML `graphs reused` counter are not evidence that CUDA Graphs were enabled.

| Native unprofiled baseline | Load (s) | Process-start visible TTFT (s) | Prefill tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 14B, microbatch 1 | 3.13 | 35.75 | 5.00 | 4.24 |
| 14B, microbatch 128 | 3.05 | 5.03 | 98.01 | 4.95 |
| 32B, microbatch 1 | 5.78 | 84.59 | 2.05 | 1.90 |
| 32B, microbatch 128 | 6.16 | 10.39 | 41.56 | 1.99 |

These are individual observations of an explicitly CPU-offloaded native application,
not tuned performance claims, xVRAM speed measurements, or all-GPU execution. RSS and
whole-device GPU samples are retained separately in the evidence. No tensor-level
reference comparison is inferred from equal generated text.

All eight direct native runs (four baseline, four CUPTI) exited zero and their owned
process trees drained. Per-model sanitized visible-output digests agree across both
microbatches and profiling/no-profiling runs. The four CUPTI traces contain respectively
744,784 / 268,636 / 743,505 / 310,678 records, with 26,860 / 8,776 / 26,758 / 9,868 GPU
kernel activities. Runtime and Driver launch callbacks are different observation layers,
not additional GPU launches; callback copy-byte sums are not PCIe traffic or VRAM usage.

Two separate Nsight commands also completed and their child trees drained. Their direct
command exit status describes the profiler, not an independently confirmed native target
exit; target stdout and TTFT are not available in these captures. The SQLite aggregates
reconcile to 8,776/9,868 kernel activities for 14B/32B. Both contain a driver-version
support warning and legacy software-instrumentation diagnostics. Nsight aggregates do
not repair missing CUPTI teardown, opaque cuBLAS arguments, or unknown per-launch bounds.

The [compact acceptance manifest](acceptance/phase6b0-20260906.json) preserves hashes and
baseline values. Full local evidence is in `artifacts/phase6b0-rtx3070-20260906`:
original `matrix.json`, ten case directories, and `review/` containing four derived
strict CUPTI reports, two updated Nsight summaries, and `review.json`. The review's
SHA-256 is `8daab2a738206af7a3ac57e5e9787c2b3352bf229e73c7b2045072fff34c30c1`.
Original artifacts were hash-checked before and after review and were not rewritten.
The review explicitly corrects the initial matrix's Nsight wording: profiler-command
success must not be presented as confirmed native-app exit.

The final analyzer intentionally does not certify typed binding declarations merely
because a source file hash matches. Observed module/function ABI, tensor subranges,
aliases/indirect accesses, complete ordering, scratch, and budget admission still need
verified contracts. These limitations prevent GO, not just performance acceptance.
Stock-process lifecycle segmentation and terminal collector completeness also remain
unproved. Resolving them does not authorize backend replacement or interposition here.

To collect a new, separately timed matrix and derive a review without changing originals:

```text
python scripts/run-phase6b-audit-matrix.py --dependencies D:/xVRAM-dependencies/phase6b0 --collector C:/path/to/collector.dll --nsys C:/path/to/nsys.exe --output artifacts/new-audit
python scripts/review-phase6b-audit-evidence.py --matrix artifacts/new-audit/matrix.json --dependencies D:/xVRAM-dependencies/phase6b0
```

The matrix harness bounds its CLI child trees as well as each native capture. Review
is offline and creates a fresh `review/` directory; it reuses captured inventory and
checks model metadata, without presenting that metadata pass as a new weight-payload
hash verification. Raw `.nsys-rep`/SQLite files remain local diagnostic artifacts.
