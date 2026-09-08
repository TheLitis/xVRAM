# Pinned kernel source-range models

These internal audit helpers are not a residency implementation or a completed
memory-safety proof. They model a bounded subset of the original llama.cpp
`b10819` source at commit `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae`.
Every model result explicitly leaves runtime binding, memory bounds, ordering,
and admission unproven. Unsupported geometry or missing evidence fails closed.

## Capture interface

`compat_audit_kernel_capture_catalog.json` contains 39 exact observed symbols:
35 source-declared argument layouts and four explicitly unsupported opaque
cuBLAS layouts. Layout offsets and sizes are independently compared with the
captured parameter bank. This comparison does not prove source-to-cubin identity.
`compat_audit_kernel_templates.template_profile()` supplies explicit reviewed
template constants keyed by catalog ID, not inferred from arbitrary symbol text.

The native observer must preserve:

- launch correlation, context/device, stream ID, launch geometry, dynamic shared
  bytes, module/function identity and parameter-layout identity;
- typed scalar fields and typed pointer members, including pointers nested in
  by-value structs; never serialized raw parameter bytes or padding;
- pointer resolution to allocation ID, lifetime generation, byte offset and
  alignment, with mapping/access segments for the actual context and device;
- an independently established tensor/subrange description, not just allocation
  size; the full model-produced intervals must fit that description and the live
  accessible mapping;
- index content generation, producer completion and absence of intervening writes
  for every indirect read. Reading an argument pointer does not provide its data.

Generate the native constexpr descriptor header without a GPU:

```text
python scripts/generate-audit-kernel-header.py --catalog python/xvram/compat_audit_kernel_capture_catalog.json --output build/audit-kernel-catalog.hpp
```

The generator hash-pins the catalog and includes `typed_capture.hpp`. It expands
struct members individually. In particular, fusion parameters occupy 48 bytes,
softmax parameters 128 bytes, and RoPE correction dimensions eight bytes with
four-byte alignment. The generated header contains `kernels` and
`catalog_sha256` in `xvram::launch_probe::typed_capture`.

## Non-obvious source requirements

- RoPE has no destination-row guard: the launch row count must match an
  independently known tensor extent. Fused row indices need content witnesses.
- `set_rows` and `get_rows` need witnessed index values and generations. Sparse
  intervals are evaluated from those values, not an allocation-size assumption.
- The observed broadcast binary kernel reads its packed additional operand at
  argument 22; the formal argument named `src1` at ordinal 1 is unused.
- Two-column quantized matrix-vector kernels can read a padded weight row even
  when the corresponding result write is guarded.
- MMQ vector-tile loads are lane-rounded and unguarded. A 40-column tile reads
  1,536 four-byte entries (6,144 bytes), not 1,440 entries. The final tile still
  requires its padded reads. These bytes need actual lifetime/access evidence.
- Stream-K fixup requires the matching scratch-producing launch and content
  generation. Scratch allocation capacity alone cannot establish initialization.
- Batched-pointer construction writes pointer tables but does not read operand
  data. Consumers need a separate matrix-range and pointer-table-generation proof.

The four opaque cuBLAS kernels cannot receive an argument ABI from guessed
private structs. A possible audit boundary is a documented public cuBLAS call,
with typed matrix dimensions, leading dimensions, scalar/pointer modes,
batch-pointer contents, workspace lifetime, and exclusive correlation to its
private launches. That boundary and its source/binary linkage remain to be
established; this module does not approve it.

## Source provenance and validation

The existing source profile is unchanged. The separate
`compat_audit_kernel_source_profile.json` pins ten additional reviewed upstream
files. Its provenance explicitly does not claim complete build-dependency
closure. `SourceKernelRanges` verifies required source bytes before evaluating
models; direct pure functions remain useful for no-driver tests only.

Run the portable model, catalog and generator tests with the repository's
`python` directory on `PYTHONPATH`:

```text
python -m unittest discover -s tests/python -p "test_compat_audit_kernel*.py"
```

No successful unit test substitutes for native bounds, source-to-cubin binding,
execution ordering, or complete trace coverage on the declared hardware matrix.
