# ADR 0001: CUDA Driver API and resident-only baseline

- Status: accepted
- Date: 2026-08-29

## Context

Windows CUDA managed memory does not provide the recoverable, fine-grained GPU demand
paging needed for a transparent oversubscribed heap. CUDA VMM does provide separate
virtual-address reservation, physical allocation, mapping, and access control.

## Decision

The core will use the CUDA Driver API and a resident-only launch contract. Host VMM is
an optional backend. The compatibility backend stores canonical data in pageable system
RAM and transfers through a bounded pinned pool.

The Driver API is loaded dynamically in the probe and distributable runtime components.
This keeps diagnostics usable without a CUDA Toolkit and allows unsupported driver
features to be reported at runtime.

## Consequences

- Integrations must declare conservative working sets before launch.
- Operations with a working set larger than the physical target require tiling or
  transformation.
- Stable virtual addresses are achievable, but mapping changes require explicit event
  boundaries.
- The project can optimize multiple frameworks without binding the core to one of them.
- Generic unchanged executables remain a later interception/instrumentation problem.
