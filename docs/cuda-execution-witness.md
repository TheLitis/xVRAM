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
WDDM resource-lifetime observation is a candidate check, not yet a hardware fact.
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

The current Windows Application Control policy blocks required local test
dependencies. Code Integrity event 3077 identifies a signing-policy failure:

- Installed PyTorch `c10_cuda.dll`, SHA-256
  `d53bedb567148250d4d63460ff703a77a26462f2d2cb7ec04bac6291a4fc8018`.
- Newly compiled memory-observer callback test, SHA-256
  `fa613a5c8234c669ae64102259968a06a27a3af247c946b57e634df9a669c18f`.
- Typed-capture test after compiling the generated 39-kernel table, SHA-256
  `068327cd91e007b291d4a63cb7d45ca99e9991c79ec6da8aab55d67f78f31fe6`.

The two native tests compiled with warnings-as-errors but these blocked
executions are **not** counted as passed. The pre-generated-table typed-capture
test, pure allocation/VMM model and post-mortem child tests did
execute. No security-policy, UAC, driver, registry, TDR or NVIDIA-setting change
is part of this work. A permitted test environment is needed before claiming the
blocked native and PyTorch regression gates.

The Debug build completed. Its initial CTest run passed 81/83 cases: one failure
was the blocked PyTorch import, and one consumer build lacked the Visual Studio
developer environment. The latter passed when rerun with that environment; it
is not a product failure. The audit suite, typed capture, and frozen ABI/schema
checks passed. A full green local regression gate is still not claimed.
The later focused audit/frozen-contract run passed 9/9 before that final
generated-table rebuild; after the rebuild the policy blocked the new typed-test
binary. No attempt was made to evade the block by renaming or repackaging it.

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
