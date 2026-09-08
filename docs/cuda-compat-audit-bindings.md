# Phase 6b.0c: observed module identity, not transparent execution

## Result — bounded pilot concluded NO-GO

The 2026-09-08 RTX 3070/WDDM pilot ran the unchanged pinned llama.cpp b10819
completion frontend and CUDA backend with Qwen2.5-14B-Instruct Q4_K_M,
microbatch 128, context 2048, FP16 KV, eight GPU-offload layers and 32 generated
token limit. This is an ordinary **CPU-offload diagnostic run**, not xVRAM
oversubscription or a speed comparison. Native execution exited zero and its
owned process tree drained. Driver 616.56 was observed, not installed or changed.

| Observation | Pilot result |
| --- | ---: |
| Strictly validated v3 trace records | 268,657 |
| GPU kernel activities / reconciled | 8,776 / 8,776 |
| Observed loaded modules / hashed | 21 / 21 |
| Hash-and-size matches to pinned static cubins | 16 |
| Function activities | 0 |
| Native `cuModuleGetFunction` lookups | 0 |
| Proven launch-to-cubin bindings | 0 |
| Owned cubin bytes copied / peak active | 43,784,544 / 27,292,936 |
| Pending copies / active copy bytes at footer | 0 / 0 |

Every submitted module copy retired with a hash record. Drop, collector error,
serialization error and incomplete activity counters were zero. However,
141,016 callback details remained unknown and stock-process finalization remained
`process_exit_unflushed`, with `complete:false`. Empty operational diagnostics do
not erase these explicitly reported evidence limitations.

The sixteen matches establish that those exact cubin bytes were loaded. They do
not establish which module supplied any particular launch. The remaining five
loaded hashes are outside this pinned backend's static inventory; they are not
assigned an owner by guessing a library or matching a kernel name.

The pilot reached the approved structural stop: the current kernel activity has
no module/function ID, function activities were absent, native function lookups
were absent, and the collector has no documented native-to-CUPTI module bridge.
The three other matrix cases were therefore **not run**. No hooks, extra CUDA
queries, backend rebuilds or kernel replacements were added to bypass this gap.
This result is not a proof that transparent compatibility is impossible.

## Observation and evidence model

Trace v3 is explicit opt-in (`--trace-version 3`); v1 stays the default and v2
keeps its prior behavior. During module-load resource callbacks, the collector
copies only the cubin bytes supplied by CUPTI into owned CPU memory. Its CPU
flusher computes SHA-256 outside callbacks, frees the copy, retires its credit
and emits `module_hash` with the original resource sequence and generation.
The documented lifetime boundary is [CUpti_ModuleResourceData](
https://docs.nvidia.com/cupti/api/structCUpti__ModuleResourceData.html).

Admission is bounded at 64 MiB per cubin, 128 MiB simultaneously owned including
hashing, and 512 MiB cumulative copies. Queue exhaustion never waits for space.
OOM, missing hashes, bad lifetimes and write errors make evidence incomplete.
No cubin bytes, CUDA addresses, native handles or application data are serialized
or saved as binary files. Existing CUDA callback observation remains unchanged
for v1/v2; v3 introduces no CUDA submissions or additional CUDA API queries.

Native module/function IDs are context-scoped with lifetime generations. CUPTI
module IDs remain a separate namespace. Buffered function activity preserves
`functionIndex` but reports an unknown generation, because arrival order cannot
select a reused lifetime. It is not assumed to be an ELF symbol-table index.

The offline command validates the static report and consumes bounded v3 records,
checks resource/hash links and successful native lookup/launch lifetimes, reuses
the independent Runtime/Driver/GPU correlator, and rechecks consumed input hashes.
It can report observed native lookup chains, but deliberately has no promotion
rule for the missing native/CUPTI bridge. Names, timestamps, nested calls and
unique candidates never manufacture that rule. Proven launch bindings and
execution readiness remain false in this observer's report contract.

`input_valid` is conservative: it includes trace completeness limitations, not
just JSON syntax. The pilot's false value coexists with schema-valid input and
fully reconciled *observed* launch records. This is not contradictory evidence.

## Reproduction and contracts

Use the existing hash-pinned binaries/models and installed CUPTI described in
the [main audit guide](cuda-compat-audit.md). The collector is built as host C++
with warnings-as-errors. Do not modify the installed driver/toolkit or run this
simultaneously with another profiler.

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
$env:CUDA_PATH = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3'
python -m xvram.compat_audit --stage all --trace-version 3 `
  --model 14b --microbatch 128 --gpu-layers 8 `
  --binary-dir D:/xVRAM-dependencies/phase6b0/llama-b10819 `
  --model-dir D:/xVRAM-dependencies/phase6b0/qwen14b `
  --capture-mode cupti `
  --collector build/compat-audit-collector/Release/xvram_compat_audit_collector.dll `
  --output-dir artifacts/new-binding-pilot --no-text

python -m xvram.compat_audit_bindings `
  --trace artifacts/new-binding-pilot/cupti-trace.jsonl `
  --binary-evidence artifacts/phase6b0-binary-evidence-20260906/binary-evidence.json `
  --json artifacts/new-pilot-bindings.json

python scripts/validate-compat-bindings.py `
  --trace artifacts/new-binding-pilot/cupti-trace.jsonl `
  --binary-evidence artifacts/phase6b0-binary-evidence-20260906/binary-evidence.json `
  --bindings artifacts/new-pilot-bindings.json
```

The command creates a fresh result; it never overwrites existing evidence. Its
worker runs under the existing Windows Job Object/Linux process-group controller,
with a 300-second total deadline (configurable 1–900 seconds). Timeout or invalid
worker output produces a schema-valid fail-closed report. Scratch is removed only
after confirmed process-tree drain. Worker exit and final report must agree.
Exit zero means **analysis completed**, not GO. Other exits are 23 invalid input,
26 timeout, 27 worker/platform/cleanup failure, 64 usage, and 74 output I/O.

New contracts are `xvram.cuda_compat_audit_trace` v3 and
`xvram.cuda_compat_launch_bindings` v1. All previous production ABI/schema/export
contracts, audit v1/v2 schemas and static binary evidence v1 remain byte-identical.
The new CLI is included in the installed Python package and no-driver CI.

## Validation and next boundary

Local checks passed: Release 85/85 CTest, serial Debug 81/81 CTest, 283 focused
Python tests (four platform skips), standalone collector/core build, installed
Python CLI/contract smoke, all 268,657 trace records against JSON Schema, and both
offline reports against JSON Schema and their semantic contracts.

The [acceptance record](acceptance/phase6b0c-bindings-20260908.json) pins collector,
input, source and result hashes. Full artifacts remain local under
`artifacts/phase6b0c-bindings-20260908`. Version remains `0.1.0-dev`.

The next step requires a separately approved observation/design boundary capable
of establishing the missing identity edge. This implementation does not authorize
one. Typed arguments/ranges, native scratch and VMM aliases, device ordering,
complete finalization and live working-set admission remain subsequent gates.
