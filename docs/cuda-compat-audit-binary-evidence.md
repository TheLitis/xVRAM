# Phase 6b.0: static binary layouts and source-only memory models

## Result and boundary

The offline follow-up establishes static device parameter layouts for **35 of 39
observed kernel names**, using the pinned official `ggml-cuda.dll` and NVIDIA
`cuobjdump`. It also adds checked source-only arithmetic for ordinary q8 quantization,
MMQ stream-K fixup allocation, and chunk-rounded range unions. **NO-GO remains.**

This is evidence refinement, not interception or an oversubscription run. No production
residency code, native collector, application/backend, CUDA settings or Phase 1–6a
ABI/schema/export contracts changed. No new GPU workload ran for this follow-up. Its
four input reports are the earlier, explicitly CPU-offloaded native observations.

Static function names and parameter offsets do not establish the identity or argument
values of a captured launch. The report deliberately cannot claim execution readiness,
complete profile coverage, memory-bound proof or xVRAM oversubscription. Exit `0`
means the bounded **offline inspection completed**, not that the audit reached GO.

## Binary inspection

`python -m xvram.compat_audit_binary` runs a separate worker inside the existing audit
controller's Windows Job Object/Linux process-group boundary. Before invoking a native
tool, that worker verifies the fixed executable and backend hashes, reads bounded input
reports and validates their consumed provenance/name/counter fields. Full input audit
report/trace validation remains the separate frozen-contract gate.

This tool profile is Windows CUDA 13.3.73 and `sm_86`, not an arbitrary-toolchain promise:

| Input | SHA-256 |
| --- | --- |
| Official b10819 `ggml-cuda.dll` | `88350839e27a43212a52cf6686562b2b6ef1498c59391dfb8cda49f3ebd89a62` |
| Installed CUDA 13.3.73 `cuobjdump.exe` | `cb72e7e353ed20320deb4770f75291a828c0b23ca8c2d27688c4eeced96543f5` |

The pinned upstream commit is `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`.
The tool lists embedded images, extracts only `sm_86` cubins into a fresh external
temporary directory, and checks exact agreement between the listed and extracted files.
The parser validates ordinary ELF64 little-endian CUDA headers, sections, string/symbol
tables, executable function sections and code spans. It reads CUDA-specific parameter
metadata from NVIDIA's textual dump, not from guessed private numeric ELF attributes.
See the official [CUDA Binary Utilities guide](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
for the inspection/extraction tools; their text is treated as a pinned tool format,
not a stable public CUDA argument ABI.

For each selected function the parser requires a single parameter-bank extent,
contiguous parameter ordinals, bounded sizes, nonoverlapping offsets and exact terminal
extent. ABI padding is allowed. Module hashes, function symbol/section indices and
raw-dump hashes are retained; raw dumps, local paths, CUDA addresses and native handles
are not serialized. Parameter sizes do **not** imply pointer types, structure members,
read/write directions or indirect tensor ranges.

Limits include 16 input reports, 4,096 distinct names, 1,024 selected cubins, 64 MiB per
cubin and 512 MiB total extracted data. Tool I/O is incremental and bounded: 64 KiB per
line, at most 128 MiB of stdout per ELF dump, and 30 seconds per tool invocation.
The outer worker deadline defaults to 300 seconds (maximum 900), also covering input
I/O, hashing and parser work. This offline command has a total deadline, not a fabricated
GPU-progress heartbeat. The existing capture controller owns descendant termination/reap.

A tool must exit normally and reach both pipe EOFs; a retained descendant writer,
truncated output, unexpected stderr or parser failure cannot yield successful evidence.
The controller validates the worker report and exit code before trusting it, overwrites
cleanup from its own observations, and removes only its exact fresh scratch directory
after confirmed tree drain. Unconfirmed containment retains that directory and fails
closed. Existing evidence output is never overwritten.

The separate strict [schema](../schemas/cuda-compat-audit-binary-evidence-v1.schema.json)
is `xvram.cuda_compat_binary_evidence` v1. The dependency-free semantic validator
`xvram.compat_audit_binary_contract.validate_report` additionally reconciles all module,
candidate, layout, activity and tool counters. Completed reports require confirmed
cleanup and empty diagnostics; failures and timeouts retain NO-GO and honest cleanup.
Exit codes are `0` completed, `23` unsupported/input preflight, `26` timeout, `27`
tool/worker/protocol/cleanup failure, `64` usage and `74` output I/O.

## Source-only memory arithmetic

`SourceMemoryRules(source_root)` in `xvram.compat_audit_memory` verifies five relevant
files from the existing 18-file hash-pinned source profile before applying a rule.
Every result carries source hashes/line anchors and explicitly states
`source_model_only`, `runtime_binding_proven=false`, `device_ordering_proven=false`,
and `admission_ready=false`. Inputs are declarations, not decoded live arguments.

- Ordinary `quantize_q8_1`: positive dimensions, nonnegative F32 element strides,
  valid/padded columns, 32-value/36-byte q8 blocks, exact declared grid and a warp-aligned
  one-dimensional block. The input bound is a conservative strided read envelope;
  output includes complete padded blocks. Broadcast read aliases are allowed, input/
  output overlap is not. The compiled block limit and correct fast-division descriptor
  still require invocation proof; no unpinned wrapper constant is inferred.
- NVIDIA MMQ stream-K: conditional float fixup payload from explicitly declared tile
  geometry, SM count and Q4_K/Q6_K shape. Integer intermediates and the source's linear
  index bound are checked. Native VMM suballocation rounds to 128 bytes; physical pool
  growth remains unknown without pool state/granularity. This is not a bound for MMQ
  operands, shared memory, other scratch or cuBLAS workspace.
- `chunk_rounded_union`: checked relative ranges within allocation generations, with
  explicit base modulo chunk alignment. It unions aliases inside one allocation, rejects
  contradictory generations/layouts and overflow, and does not assume native allocations
  are chunk-aligned. Distinct allocations are counted separately: without shared-reservation
  alias proof their sum is conservative, not exact physical residency.

These functions are not connected to runtime admission. A plausible scalar or allocation
size cannot be promoted to a tensor-bound proof by passing it through checked arithmetic.

## Reproduction

Use the existing pinned dependencies and four reports from the
[earlier refinement](cuda-compat-audit-refinement.md). Keep extraction outside the source
tree and choose a new JSON path:

```powershell
$env:PYTHONPATH = 'python'
python -m xvram.compat_audit_binary `
  --backend D:/xVRAM-dependencies/phase6b0/llama-b10819/ggml-cuda.dll `
  --cuobjdump 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/cuobjdump.exe' `
  --report artifacts/phase6b0-refinement-20260906/14b-ub1/report.json `
  --report artifacts/phase6b0-refinement-20260906/14b-ub128/report.json `
  --report artifacts/phase6b0-refinement-20260906/32b-ub1/report.json `
  --report artifacts/phase6b0-refinement-20260906/32b-ub128/report.json `
  --work-root D:/xVRAM-dependencies/phase6b0/binary-audit-work `
  --json artifacts/new-binary-evidence.json
```

The Python package and CPU tests run without CUDA or PyTorch on Windows/Linux. The
actual pinned binary/tool gate here ran on Windows; a Linux-native cuobjdump profile
has not been accepted. The command does not install or alter a CUDA Toolkit.

## Recorded offline results — 2026-09-06

The complete inspection exited `0`, passed JSON Schema and semantic validation, reported
empty diagnostics and confirmed worker reap, tree drain and scratch removal. It scanned
143 cubins (111,265,920 bytes; 6,709 function symbols). Sixteen modules contained the
selected names; 19 tool invocations consumed 50,037,405 stdout bytes. All 72,262 input
activity counts reconcile. Of 39 names, 35 have one static-layout candidate, four are
missing from this backend, none are ambiguous, and **zero have captured-module bindings**.

| Example kernel family | Device parameters | Parameter bank bytes |
| --- | ---: | ---: |
| `quantize_q8_1` | 9 | 72 |
| `quantize_mmq_q8_1` | 11 | 76 |
| `mul_mat_vec_q` | 19 | 160 |
| `mul_mat_q` | 24 | 172 |
| `mul_mat_q_stream_k_fixup` | 13 | 100 |

These are device parameter-block layouts, not host launch-wrapper structure sizes.
The four missing names are cuBLAS/CUTLASS library kernels. Their private ABI is not
guessed: memory contracts must instead be established at the public operation boundary.
The [acceptance record](acceptance/phase6b0-binary-evidence-20260906.json) records report,
tool/backend and implementation hashes, cleanup and local validation.

Local checks passed: 85/85 Release and 81/81 serial Debug CTest targets; 265 focused
Python tests with four POSIX-only skips on Windows; installed Python CLI/contract/source
manifest smoke; and unchanged frozen ABI/schema/export checks. Audit discovery stays
separate from the 114-test general Python suite, with the existing 90/60-second limits.
CI installs and smoke-tests the new offline command while keeping all existing jobs.

## Remaining gates

1. Bind an observed launch to its exact loaded module/function. The current GPU kernel
   activity record has no module/function ID; the current resource trace has module ID
   and cubin size but no cubin hash. No function activities were emitted in the four
   input traces. A matching static name alone cannot bridge this gap.
2. Establish typed scalar inputs, pointer/alias/indirect ranges and native scratch/pool
   lifetimes, including library calls, without reading unknown argument layouts.
3. Prove device ordering, complete collection/finalization and live chunk-rounded
   CUDA/WDDM admission. The previous incomplete terminal evidence and Nsight
   driver-support warning have not disappeared.

Any future observation extension must remain bounded and read-only; borrowed callback
data cannot outlive the callback without an owned copy. No CUDA submissions, mapping,
argument mutation, exit hooks, private-library ABI guesses or backend replacements are
authorized by this offline result. Resolve these gates before approving Phase 6b.1.
