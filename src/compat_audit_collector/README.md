# Diagnostic CUPTI collector (Phase 6b.0)

This is a host-C++ profiling library, **not a CUDA compatibility adapter**. It
does not change allocations, arguments, streams, return values, CUDA functions
or kernels. It makes no CUDA runtime/Driver API calls or GPU submissions. Its
only GPU-related dependency is NVIDIA CUPTI. Tracing overhead is not application
performance evidence.

### Opt-in resolver observations (trace v2)

The default remains the byte-compatible v1 record format. Set
`XVRAM_AUDIT_TRACE_VERSION=2` for a separate v2 trace; unset or `1` selects v1,
and `3` selects module observations; other values refuse initialization. Version 2 observes fixed
public metadata for `cuGetProcAddress`, `cuGetProcAddress_v2`,
`cudaGetDriverEntryPoint` and `cudaGetDriverEntryPointByVersion` (including their
runtime `_ptsz` variants). No resolver result is changed or invoked.

The optional fields are `requested_symbol` (bounded public C identifier),
`requested_version` when present in the API, `resolver_flags`, and, only after
successful API return, `query_status` when its output argument is supplied and
`entry_point_id` when resolution returned a non-null entry point with successful
or unavailable query status. The entry point is converted into a bounded
process-local identity before serialization. Neither output pointer storage nor
function code is serialized. Failed calls never have their out-parameters read.
The v2 metadata still has `op:"other"`: observing resolution does not establish
the function's invocation, owning binary, or semantic contract.

`xvram.compat_audit_routing.summarize_routing` derives bounded request/status
counts from one trace, while preserving Runtime/Driver callback layers as
separate observations. Duplicates, incomplete pairs, changed inputs, unknown
metadata, missing results and incomplete terminal evidence stay unresolved.
`routing_coverage_complete` and `semantic_ranges_proven` are always false. CUPTI
resolution callbacks do not observe Windows `GetProcAddress`, private export
tables, every registration path, or the use of a returned pointer. Trace v2 does
not change terminal-flush limitations below and cannot independently produce GO.

## Build and loading

### Opt-in module identity observations (trace v3)

`XVRAM_AUDIT_TRACE_VERSION=3` retains resolver metadata and adds module load/unload
generations, successful native function lookups and bounded owned cubin copies.
The default remains v1; v1/v2 schemas and serialization are unchanged.

Only `CUpti_ModuleResourceData::pCubin` is copied, during its load callback. The
CPU flusher hashes each owned copy outside callbacks and emits `module_hash`
with its original resource sequence, context/module IDs and generation. Copies
are limited to 64 MiB each, 128 MiB active (including hashing), and 512 MiB total.
There is no waiting for queue space. OOM/limit/write errors make evidence incomplete.
Raw cubins are neither serialized nor written to disk. Exit remains unflushed;
pending copies and hash counts appear in the best-effort incomplete footer.

Native module/function namespaces remain separate from CUPTI IDs. Buffered
function activities retain `functionIndex` but report module generation zero:
arrival order cannot select a potentially reused generation. The symbol index
is not presumed to be an ELF symbol index. Module hashes, names, temporal nesting
and unique candidates do not establish the missing launch/module bridge. V3
adds no CUDA calls, private-ABI decoding, exit hooks or kernel instrumentation.

The collector requires an installed CUDA Toolkit/CUPTI 13.3 or newer, a 64-bit
host compiler, and C++20. Windows x64 is the primary build. The POSIX file backend
also permits a Linux build; Linux hardware tracing is not claimed as validated.

Build this directory directly with CMake, or enable the root project's optional
`XVRAM_BUILD_COMPAT_AUDIT_COLLECTOR` option. `XVRAM_CUPTI_ROOT` selects the official
CUPTI installation. The target is `xvram_compat_audit_collector`. CPU-only registry
and serialization tests are controlled by `XVRAM_COMPAT_AUDIT_COLLECTOR_TESTS`.
No CUDA language compiler or cudart/cuBLAS/Driver link is needed.

For a separately authorized diagnostic launch, set `CUDA_INJECTION64_PATH` to
the collector's absolute DLL/shared-library path and `XVRAM_AUDIT_TRACE` to a new
JSONL output filename. The CUDA loader calls the exported `InitializeInjection`.
The CUPTI runtime must be available through the platform's normal library search
path (on this Windows Toolkit, `extras/CUPTI/lib64`). The collector exclusively
creates the output file and refuses to overwrite an existing file. Do not set
`NVTX_INJECTION64_PATH`; this collector does not implement NVTX injection.

## Completion is deliberately conservative

The installed NVIDIA injection sample warns that Windows `atexit`, static
destructors and DLL detach cannot safely flush CUPTI, and uses a Detours process
exit hook. **This collector does not include that hook.** A dedicated CPU thread
requests completed activity buffers every 200 ms while the application is alive.
An ordinary exit attempts a nonblocking, CPU-only summary with `complete:false`
and `terminal_checkpoint:"process_exit_unflushed"`. No final footer is guaranteed
after abrupt termination, loader teardown, or a busy/terminated writing thread.
Missing, partial or incomplete terminal evidence is always NO-GO; a separate
Nsight capture does not upgrade this collector's completeness.

Controlled test harnesses may call the exported C function
`int XvramAuditFinalize(void)` **before CUDA teardown**, after the harness itself
has synchronized all submitted CUDA work and stopped every thread from entering
CUDA again. Do not call it from a CUPTI/CUDA callback or a stream host callback.
It joins the flush worker, disables activity collection, force-flushes CUPTI,
unsubscribes and writes a terminal summary. It does not synchronize CUDA itself.
The library must remain loaded through process termination; runtime unload is
unsupported. Finalizing ends observation: any later CUDA calls invalidate the
external harness's proof. The function's return indicates finalization mechanics,
not interoperability approval; inspect the summary and analyzer report.

`complete:true` additionally requires zero drop/error/unknown-detail counters,
matched entry/exit balance, complete activity timestamps and returned activity
buffers. The initial collector intentionally reports many API argument details
as unknown; observing all callback names is not proof of complete API semantics.

## Trace contract and bounds

Every line has `schema_version:1`,
`report_type:"xvram.cuda_compat_audit_trace"`, a one-based monotonic `sequence`,
`kind`, and `timestamp_ns`. Envelope timestamps use `steady_clock`; activity
start/end timestamps retain CUPTI's separate clock domain. Kinds are `session`,
`api_enter`, `api_exit`, `activity`, `resource`, `gap`, and `summary`.

Callbacks include domain, callback ID, correlation ID, process-local thread ID,
symbol, exit status, `detail_known`, and available fixed API argument metadata.
`parameter_bytes` is the size of the **CUDA API metadata structure**, never the
number/size of a kernel's arguments. Launch geometry and function/name identities
are observations, not tensor access declarations. No kernel argument arrays,
application data, module binary bytes, raw virtual addresses, or raw handles are
serialized. Numeric native values are confined to a bounded in-process registry.

Allocation IDs and generations describe successfully observed API allocations;
copy ranges are checked against these API allocation extents. They are **not
tensor bounds**. Runtime/Driver duplicate descriptions of an identical live
allocation share an ID; conflicts and overlaps produce incomplete evidence.
Freed/reused allocations receive new IDs/generations. CUPTI stream IDs are scoped
by context. Native event/physical-handle retirement prevents identity reuse.
CUPTI/native module/function ID namespaces remain distinct unless an independent
semantic contract proves the relationship. Delayed allocation activities are
not assigned a potentially reused address's current allocation ID.

Recorded fixed details currently cover common allocations and pinned host
registration, 1D/2D runtime copies, runtime memsets, native VMM operations, kernel
launch geometry and selected stream/event operations. All other enabled
runtime/driver APIs still produce paired callback records with unknown details.
Resource callbacks and kernel/API/copy/memset/memory/function activities are also
recorded. cuBLAS API visibility is explicitly unavailable: CUPTI runtime/driver
observations are not cuBLAS semantic calls. Unknowns must not be silently allowed.

Limits are 1 MiB per serialized record, 4096 input bytes per string, 65536 total
native registry slots, 65536 thread IDs, one million nonterminal records, and at
most sixteen one-MiB CUPTI activity buffers. Limit exhaustion and failed writes
increment dropped/error counters. Callback exceptions never cross the CUPTI ABI.
The JSONL file itself can still be large; the controller must impose run duration,
disk and host-memory limits. No pointer-like exception/error message is emitted.

## Source references

The implementation was developed after reading the complete installed Toolkit
13.3 `extras/CUPTI/samples/cupti_trace_injection/README.txt` and
`cupti_trace_injection.cpp`. It uses the documented profiling entry point,
subscriber/activity callbacks and periodic/explicit flushing, but does not copy
the sample's API/exit hooks, CUDA work, NVTX payload inspection or raw-address
printing. Official declarations were checked in `cupti_callbacks.h`,
`cupti_activity.h` and the generated CUDA API metadata headers supplied by CUPTI.
See [NVIDIA CUPTI documentation](https://docs.nvidia.com/cupti/main/main.html).
