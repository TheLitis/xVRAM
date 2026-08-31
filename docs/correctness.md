# Correctness and failure policy

Performance policy is replaceable; correctness policy is not.

## Launch protocol

For every managed operation the runtime:

1. validates that the declared working set can fit the current physical target;
2. pins all ranges participating in the operation;
3. chooses only unpinned, idle victims;
4. finishes required dirty write-back;
5. establishes mappings and access rights;
6. enqueues or awaits H2D completion;
7. launches after the residency dependency;
8. records completion before ranges become evictable.

If the working set is larger than the safe VRAM target, strict mode returns a structured
error requiring tiling. It must not rely on an unrecoverable GPU access fault.

The Phase 3 known-operation planner applies the same rule to every GEMM tile. It pins
the exact A/B/C chunks and bounded library workspace before calling cuBLAS and releases
those ranges only after the post-library event generation retires. K panels for one C
tile are contiguous, and event-safe write-back/reload preserves accumulation if C does
not remain resident. If budget shrink makes the next tile impossible, it drains and
returns budget pressure; it never launches against a partial working set.

FP16/BF16 inputs accumulate in FP32. K-panel splitting is used only when C is FP32 (or
FP64 for FP64 compute); a low-precision C keeps the complete K dimension in one library
call so partial sums are never rounded through FP16/BF16 storage between panels.

## Error handling

- Every CUDA and platform result is checked and retained in the diagnostic report.
- Allocation failure triggers bounded policy recovery, never unbounded retries.
- A changing WDDM budget reduces future residency targets and may block new work.
- Device loss or a failed synchronization poisons the affected runtime instance.
- Cleanup is best effort after device loss, but no later work is accepted.
- The capability probe treats unsupported features as findings, not fatal errors.

The Phase 0 probe releases a VMM allocation handle even if its mapping cannot be
removed, as permitted by the CUDA Driver API, and quarantines any unresolved mapping or
reservation for the remaining driver lifetime.

The Phase 1 PoC adds an injectable Driver API dispatch, an explicit cleanup ledger, and
process isolation. It never unmaps or reuses a slot until its generation's completion
event reports success. If an unmap fails, the handle is still released but the original
reservation is deliberately retained; the worker exits as the quarantine boundary.
The controller kills and reaps a crashed or stalled worker without changing global CUDA,
TDR, or NVIDIA configuration.

The Phase 2 cache applies the same quarantine rule to every frame and reservation while
adding independent event generations for compute, H2D, and D2H staging. A dirty frame
cannot be reused until D2H completes and pinned bytes have been copied to pageable
backing. Policy output is advisory: the manager rejects pinned, in-flight, aliased, or
current-working-set victims. During budget shrink, new launches stop until the active
transaction is drained and excess clean or written-back frames are released.

The Phase 3 SDK serializes CUDA and cuBLAS work on a session worker thread. A generic
transaction callback is trusted in-process code, but its contract is deliberately
narrow: enqueue only on the supplied stream, do not synchronize or replace the current
context, and do not retain any resolved device address or workspace pointer. The runtime
records a completion event immediately after callback return. A non-success callback
result fails the operation; any submitted work is retired through its event generation
before the pinned ranges are released, then the session is conservatively poisoned and
accepts no later operation.

Isolated sessions own and destroy their CUDA context. Attach-current sessions borrow the
context captured during creation and only push/pop it on the worker thread. They never
destroy, reset, or retain ownership of the caller's context. Because an SDK callback can
hang inside the caller process, hard termination is intentionally available only through
the isolated `xvram-gemm-bench` controller/worker boundary.

`cuEventQuery` accepts only `CUDA_SUCCESS` and `CUDA_ERROR_NOT_READY`; every other result
poisons the worker. On Windows, a GEMM tile observation above 250 ms prevents a
subsequent tile launch and reports watchdog risk. A
failed unmap still releases the physical handle, deliberately retains the reservation,
marks cleanup incomplete, and relies on worker-process exit as the quarantine boundary.
If asynchronous completion cannot be established, the runtime also deliberately
abandons owned CUDA modules, cuBLAS handles, streams, events, pinned buffers, and the
isolated context so no implicit destroy-time synchronization can cross that boundary.

## Validation strategy

- deterministic state-machine unit tests;
- property tests over legal and illegal transitions;
- byte-pattern and checksum tests across repeated remaps;
- read-only, dirty, reuse, sequential, and adversarial access traces;
- fault injection at every CUDA/platform call boundary;
- long-running pressure tests with changing desktop load;
- comparison against a CPU reference for every proof-of-concept kernel.

No benchmark result is accepted unless correctness validation for the same configuration
passes first.

The overlap benchmark hashes its embedded PTX source, checks the synthetic kernel's first
output word against the same unsigned recurrence on the CPU, verifies the complete copy
buffer, and requires successful resource cleanup before reporting `completed`.

The VMM proof hashes its versioned PTX module, verifies every returned tile, repeats a
complete CPU comparison after each requested mode, and requires matching 128-bit
digests, event-safe remap counts, stable addresses, real handle reuse, and a fully true
cleanup ledger before reporting `completed`.

The residency-cache benchmark hashes `xvram_residency_workload_v1`, checks verification
tokens at each operation, and fully compares pageable backing with the CPU model after
each workload. CLOCK and LRU runs start from identical regenerated backing. Completed
reports additionally reconcile hit/miss, prefetch-terminal, dirty-writeback,
map/access/unmap, handle-lifecycle, target, staging, and cleanup counters.

The GEMM benchmark uses two validation paths. Small and boundary cases compare every
element against a higher-precision CPU reference with format- and compute-mode-specific
tolerances. Oversubscribed cases use deterministic structured operands whose full result
can be evaluated in O(M x N), so validation does not require a second O(M x N x K) host
GEMM. Acceptance requires finite outputs, zero out-of-tolerance elements, matching
digests for equivalent plans, event-safe map/access/unmap accounting, a bounded working
set and workspace, complete cleanup, and no residual worker. Every requested pass is
read back, fully checked, and included in the validation digest; `elements_verified`
therefore equals M x N x passes. Reported TFLOPS uses only measured library GEMM time,
while workload elapsed time remains the complete residency-and-validation observation.
