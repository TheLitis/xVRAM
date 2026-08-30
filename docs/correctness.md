# Correctness and failure policy

Performance policy is replaceable; correctness policy is not.

## Launch protocol

For every managed operation the runtime will:

1. validate that the declared working set can fit the current physical target;
2. pin all ranges participating in the operation;
3. choose only unpinned, idle victims;
4. finish required dirty write-back;
5. establish mappings and access rights;
6. enqueue or await H2D completion;
7. launch after the residency dependency;
8. record completion before ranges become evictable.

If the working set is larger than the safe VRAM target, strict mode returns a structured
error requiring tiling. It must not rely on an unrecoverable GPU access fault.

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

`cuEventQuery` accepts only `CUDA_SUCCESS` and `CUDA_ERROR_NOT_READY`; every other result
poisons the worker. A kernel observation above 250 ms prevents a subsequent launch. A
failed unmap still releases the physical handle, deliberately retains the reservation,
marks cleanup incomplete, and relies on worker-process exit as the quarantine boundary.

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
