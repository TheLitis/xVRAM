# Native identity witness and PC-sampling permission gate

Status: **partial native evidence; all four execution proof gates remain open**.
Production residency, old ABIs/exports/schemas and version `0.1.0-dev` are unchanged.

## Observed library/module edge

The optional `XVRAM_PROBE_WITNESS=ON` build requires the legacy-resolver census.
`python -m xvram.compat_launch_probe run --identity-witness ...` produces a separate
`identity.jsonl` and `identity-report.json`. Successful Driver library load/unload
outputs are copied inside their valid CUPTI callback lifetime; callbacks do not
query CUDA, submit work or change arguments. Census API instance IDs correlate the
sidecar with actual API exits, including failed loads. Library IDs and generations
are process-local and checked for stale/duplicate loads and unloads.

For each contextless launch, documented `cuKernelGetLibrary` and `cuLibraryGetModule`
queries must agree with the independently queried `cuFuncGetModule` result. The
original handle/arguments are still forwarded unchanged. A mismatch rejects the
launch and poisons the diagnostic. CUfunction-only launches are explicitly counted
as lacking this library edge. A `module_observation_id` is **not** a generation or
CUPTI module ID, and is never converted to a cubin binding by name uniqueness.

The separate v1 sidecar/report schemas reject native pointer fields. The bounded
no-driver parser checks framing, sequence, exact fields, library lifetime, call
reconciliation, census API identity, missing/error footers and input modification.
The controller now returns failure when the requested sidecar is absent or invalid,
even if the older census completed. Its exit `0` means this diagnostic observation
completed, not that any of the four proof gates is true.

On 2026-09-08, a new original 14B/microbatch-128 **CPU-offload baseline** observed:

- 8,776 launches and 8,776 identity records;
- 8,680 independently queried library/module agreements;
- 96 CUfunction-only launches; 28 library loads, no observed unloads;
- no witness errors; old census launch correlations reconciled.

This is not xVRAM oversubscription, 32B acceptance, memory-bounds proof, cubin
authentication, event-order proof, or terminal completeness.

## PC-sampling preflight: permission is checked after actual submission

NVIDIA documents serialized PC records with a kernel correlation ID and cubin CRC
in the [PC Sampling API](https://docs.nvidia.com/cupti/main/main.html#cupti-pc-sampling-api).
These could provide another identity edge, but sampled absence cannot prove absence
of execution, and serialization cannot prove the unperturbed application's ordering.
The attempted model sampler is **not shipped**: its first launch stalled and the
300-second controller deadline killed/reaped the owned job. Its development trace
was not accepted and its implementation remains a local artifact, not a build input.

`xvram-pc-sampling-probe --launch-smoke` is a separate isolated empty-kernel diagnostic.
It initializes the profiler, checks device support, configures sampling, submits a
tiny PTX kernel, synchronizes, requests data, and attempts module/context/sampler
cleanup. Merely accepting `cuptiPCSamplingEnable` is insufficient: the first trial
returned success with zero counters when app-local profiler DLLs were absent.
The target now includes hash-pinned `nvperf_host` and `nvperf_target`, both from the
same CUPTI 13.3.75 archive, verifies their actual loaded identities, and reports
sampling/data/cleanup results independently.

The fresh isolated hardware check saw device support and successful kernel launch,
sync, module unload and context destroy, but both data collection and sampler disable
returned **35 (`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`)**. This is a real permissions
barrier to this PC-sampling path, not a finding that transparent execution is impossible.
The report is `permission_required`, exit `23`; no proof/cleanup success is invented.

Use the controller rather than directly running the executable:

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
python -m xvram.compat_pc_preflight `
  --probe build/compat-identity-witness/xvram-pc-sampling-probe.exe `
  --output-dir artifacts/new-pc-permission-check
```

It pins app-local dependencies, uses a fresh directory, imposes a 30-second deadline
and owns/reaps only its child tree. It emits `xvram.cuda_pc_preflight` v1. A future
`ready` result is permission/readiness evidence only, never execution proof.
Do not run a longer model sampling attempt until this gate succeeds.

NVIDIA's [documented remedy](https://developer.nvidia.com/ERR_NVGPUCTRPERM) includes
running the diagnostic with administrative privileges. An elevated child requires
an explicit UAC interaction; the current implementation never elevates itself or
changes driver, TDR, registry, NVIDIA settings or system-wide counter permissions.

## Remaining proof work

1. **Memory bounds:** authenticated typed invocation scalars, tensor/alias lifetime,
   indirect accesses and native scratch; current formulas are conditional only.
2. **Cubin:** live binary identity for every launch, including unsampled/opaque library
   kernels; native library/module agreement is not yet the missing image edge.
3. **Ordering:** actual context/stream/event generations, waits and GPU retirement;
   conditional vector-clock tests are not hardware evidence.
4. **Completeness:** producer quiescence, GPU drain and final delivery outside callbacks
   and loader lock. Current observational footers intentionally remain false.

Both model sizes and both microbatches remain required before a GO decision. See the
[acceptance record](acceptance/phase6b0h-witness-20260908.json) for immutable evidence
hashes. No system configuration was changed during these tests.
