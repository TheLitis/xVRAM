# Native execution witnesses: incomplete, fail-closed audit

Status on 2026-09-08: **all four profile-wide proof gates remain open**. This
increment changes diagnostic tools, source arithmetic and tests, not production
residency or transparent execution. Version remains `0.1.0-dev`. The original
llama.cpp CUDA backend runs with eight GPU layers and explicit CPU offload in
these observations; none is an xVRAM oversubscription result.

## What the new observations establish

The optional standalone probe build accepts `XVRAM_PROBE_PC_WITNESS=ON`, requiring
the existing census, identity and legacy-resolver options. It adds:

- Serialized PC sampling, with separate configured/result buffers, a pinned
  app-local CUPTI/profiler dependency pair, bounded image copies, and collision
  rejection. The CRC callback's transport key is bound to a SHA-256 image; each
  sampled kernel must match an actual Driver API correlation, never a name alone.
- Typed context/stream/event observations. Analysis rejects nested mutations,
  crossed callbacks, ambiguous implicit-stream semantics, unknown event results,
  missing retirements and incomplete census inputs. Thread observation IDs are
  monotonic TLS tokens, not reusable Windows thread IDs.
- A controller-owned shared-memory post-mortem ledger. Atomic API/activity
  counters run before file locks and closed-file guards. The mapping remains
  alive through worker death; its reader requires successful reap and must
  reconcile its API count against the same capture's census.

`XVRAM_PC_WITNESS_TRACE` enables the sampler; `XVRAM_EXECUTION_WITNESS_TRACE`
enables typed ordering. Without the former, capture does not add per-kernel
sampling synchronizations. `XVRAM_AUDIT_POSTMORTEM` identifies the controller-owned
mapping; its name is not part of public reports. No CUDA call or argument mutation
is made inside a CUPTI callback. The diagnostic still forwards each original
launch exactly once when its diagnostic preconditions succeed.

A 14B/microbatch-128 exploratory sampled run observed 8,776 sampled identities
for 8,776 native launches, with zero dropped samples. All 35 custom-kernel image
hashes match their static cubin candidates; four proprietary cuBLAS kernels have
actual sampled image identities but no source-derived argument semantics. This
is a fact about that run, not coverage of another run, nor the full 14B/32B matrix.
Earlier pilots had dropped samples or missing identities and were rejected.

Sampler-owned function names can be shared with CUPTI Activity records. Their
bounded pool is now retained until the controller reaps the owned worker, avoiding
premature frees while callbacks may still reference names. An in-process sampler
footer therefore does not claim complete cleanup or terminal completeness.

## The newly demonstrated terminal gap

The post-mortem ledger rejected an otherwise successful 14B/microbatch-128
native run: after the previous final footer it observed **21 more CUDA APIs** and
one outstanding Activity buffer. A follow-up retaining post-footer rows identified
20 successful `cuLibraryUnload` calls and one successful
`cuDevicePrimaryCtxRelease`. These are real native teardown, not additional
observed kernels. Ignoring them, or replacing their counters with zero, would be
incorrect.

Evidence directories (local, immutable diagnostic artifacts):

- `artifacts/phase6b0i-execution-20260908/matrix-cap512`: exploratory sampled
  identity result; subsequent microbatch-1 capture hit the previous trace cap.
- `artifacts/phase6b0i-execution-20260908/matrix-ledger`: rejected post-mortem
  closure, despite native exit `0` and confirmed controller reap.
- `artifacts/phase6b0i-execution-20260908/matrix-ledger-tail`: the actual late API
  rows. Strict analyzers reject rows after the supposed terminal summary.

New opt-in census v3 and identity v2 increase bounded capacity for microbatch 1;
existing census v1/v2 and identity v1 contracts are not rewritten. PC/order traces
have their own new strict schemas. A larger trace limit does not repair the
terminal lifecycle gap.

The next terminal-proof work must distinguish an execution-drain checkpoint
from actual process closure, retain library/context generations through teardown,
and establish observer-channel lifetime. A successful primary-context release
does not by itself prove that the last reference was released. Independent
WDDM resource-lifetime observation is now an exploratory hardware observation,
not a completed context-destruction proof (see the follow-up below).
Absence of queue packets alone is insufficient: packets can batch kernels and
user-mode queues can submit work without a traditional kernel submission per job.

## Memory work and remaining integration

The [source-range models](cuda-kernel-source-ranges.md) and reviewed 39-symbol
capture catalog provide source-derived, overflow-checked arithmetic for the 35
custom layouts. They explicitly account for MMQ padded tile reads, matvec tails,
indirect indices, alias restrictions and stream-K scratch slots. The four opaque
library layouts are never decoded by guessing a private Params struct.

`typed_capture.hpp` validates the whole parameter ABI before dereferencing the
host parameter bank, copies bounded snapshots, separates pointer resolution from
scalar serialization, and never emits struct padding. The header generator is
pinned to the reviewed catalog. These primitives are **not yet connected to a
completed native tensor-bounds certificate**. Actual index generations, tensor
extents, scratch producer/consumer order and public cuBLAS operation contracts
remain required.

The separate new memory observer models allocation/VMM generations, aliases,
mapping and access coverage. It is device-only; host-pinned coverage is explicitly
false. Its `resolve(pointer, 1)` is not evidence that a tensor's whole access set
fits. This diagnostic observer is not a residency manager and is not a new
production transfer path.

## Verification and environment blockers

Native post-mortem tests run without CUDA/CUPTI: an owned Job Object child exercises
clean exit, sampling-pool retention, late APIs, late Activity, open callbacks,
deterministic seal races, underflow, crash, hang and mandatory reap. Portable tests
cover source arithmetic, catalog generation, strict schemas, typed field capture,
and fail-closed order/census correlation. Prior ABI/schema/export contracts remain
unchanged.

The previous Windows Application Control policy blocked required local test
dependencies. Code Integrity event 3077 identified a signing-policy failure:

- Installed PyTorch `c10_cuda.dll`, SHA-256
  `d53bedb567148250d4d63460ff703a77a26462f2d2cb7ec04bac6291a4fc8018`.
- Newly compiled memory-observer callback test, SHA-256
  `fa613a5c8234c669ae64102259968a06a27a3af247c946b57e634df9a669c18f`.
- Typed-capture test after compiling the generated 39-kernel table, SHA-256
  `068327cd91e007b291d4a63cb7d45ca99e9991c79ec6da8aab55d67f78f31fe6`.

Those two native tests compiled with warnings-as-errors but those blocked
executions are **not** counted as passed. The pre-generated-table typed-capture
test, pure allocation/VMM model and post-mortem child tests did
execute. No security-policy, UAC, driver, registry, TDR or NVIDIA-setting change
is part of this work. After the user reported removing the blocker, fresh normal
executions succeeded: PyTorch `2.13.0+cu130` imports, the generated-table typed
capture test passes, and the real typed memory callback helper passes. This is
new execution evidence, not a reinterpretation of the earlier blocked attempts.

The Debug build completed. Its initial CTest run passed 81/83 cases: one failure
was the blocked PyTorch import, and one consumer build lacked the Visual Studio
developer environment. The latter passed when rerun with that environment; it
is not a product failure. The audit suite, typed capture, and frozen ABI/schema
checks passed. A full green local regression gate was not claimed at that point.
The later focused audit/frozen-contract run passed 9/9 before that final
generated-table rebuild; after the rebuild the policy blocked the new typed-test
binary. No attempt was made to evade the block by renaming or repackaging it.

## Typed capture follow-up after the Application Control unblock

Fresh bounded 14B/microbatch-128 runs retain the pinned original executable and
CUDA backend with eight GPU layers. They remain explicitly **CPU-offload
observations**, never xVRAM oversubscription or speed measurements.

The `memory-compound-teardown-14b-mb128` run exited `0`, with owned process-tree
reap, no truncated output and no controller errors. It captured 8,776 launch/return
pairs and 207,854 typed fields: zero capture faults, zero unresolved pointers,
and 192 explicitly opaque cuBLAS kernel calls. The device-memory observer
reconciles 1,917 API pairs with zero errors/open calls and zero live allocations,
reservations, handles or mappings at its checkpoint. This does not prove tensor
subranges, host-pinned lifetimes or global terminal completeness.

Two previous observer rejections were diagnosed from actual typed inputs:
successful `cudaFree(nullptr)` forwarded to a null Driver free is an observed
no-op, and llama.cpp unmaps one contiguous six-MiB pool formed by three complete
two-MiB mappings. The latter is legal full-range unmapping, not permission to
unmap part of an original mapping. The registry now validates the entire union
before mutation and emits each original mapping generation. Gaps, partial
mappings and cross-reservation unions remain rejected by CPU tests.

Stopping all three Activity kinds after an independently successful GPU drain,
then flushing CUPTI, now leaves zero outstanding/late Activity buffers in the
post-reap ledger. The 21 late native API pairs still remain: 20 library unloads
and a primary-context release. They are not erased or counted as proof by the
old terminal validator. A new non-final typed teardown sidecar captures their
actual library/context generations. Its first hardware run also identified
eight loaded library generations without explicit observed unloads; this remains
an unresolved lifecycle obligation, not an assumed successful destruction.

An independent, owned Nsight ETW pilot retained Device, Context and HwQueue
Start/Stop events through process exit with zero xperf-reported lost events or
buffers. Exploratory correlation found two devices, three contexts and twelve
queues. Queue identity is the observed OS `ParentDxgHwQueue` namespace, not the
zeroed driver `hHwQueue` field. This first pilot lacks controller-handle-bound
process identity and a verified QPC/SQLite clock conversion. Matched Stop events
also do not carry a successful DDI return status. Consequently no terminal gate
is inferred from those counts. Raw ETL/SQLite stays private because it contains
native identities and environment metadata; only normalized IDs may be published.

The new source adapter consumes actual captured scalar arguments and launch
geometry, never derives a tensor shape from allocation length, and reports
unavailable indirect contents/companion scratch contracts explicitly. Agreement
with source-relative intervals is separate from proving that source semantics
match the sampled cubin. On the saved 14B pilot it now derives ranges for
8,040/8,776 launches, including 402 ordinary RoPE calls whose position values
affect arithmetic rather than address selection, and 192 symbolic pointer-table
producer calls. The latter produce 23,040 symbolic pointer writes; generation
validity and consumer bounds remain separate obligations. The 544 remaining
index-dependent launches and 192 opaque library launches are explicit gaps.
The full 14B/32B, microbatch-1/128 matrix and all four
profile-wide proof gates remain incomplete.

The post-unblock Release regression with these new native helpers passed 89/89
CTest cases, including Stable-ABI loading and frozen ABI/schema/export checks.
Consumer negative-link cases now use independent build directories: a transient
Windows lock on the just-executed positive-control image must not replace the
expected unsupported-symbol diagnostic. No retry or weakened rejection rule
was added.

## Official contracts used

- [CUPTI PC sampling](https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__API.html):
  actual sample/cubin correlation and shared function-name lifetime.
- [CUPTI Activity](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html):
  stopping collection and delivering buffers are separate from GPU completion.
- [CUDA libraries](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html)
  and [primary contexts](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html):
  teardown operations and reference-dependent context reset.
- [Windows user-mode work submission](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/user-mode-work-submission):
  why missing traditional queue packets cannot prove absence of work.

No global GO, transparency, numerical inference parity, speedup, or completed
32B audit matrix is claimed by this increment.
