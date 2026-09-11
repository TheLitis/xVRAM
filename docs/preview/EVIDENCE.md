# What this preview demonstrates

## Three evidence levels — do not combine them

**Implementation:** public source, frozen report schemas and the installed SDK API.
**Automated checks:** build, no-GPU tests, negative cases, sanitizers, packaging and
relocation in the CI runs linked by the release. **GPU observations:** maintainer
historical hardware results, or a new recorded run of a specific packaged binary.
A green GitHub Actions matrix does not mean the runner had an RTX 3070.

The package manifest records `gpu_validation: not_run_by_packaging_ci`. Launch
preparation did not create new GPU measurements, recover local-only artifacts from
the maintainer's PC, or convert unit-test fixtures into hardware results.

## Historical baseline, not a measurement of the new package

These values were documented in the repository at `39e2a3b`. Their source is the
maintainer's [Phase 6a record](../cuda-compat.md#recorded-local-results), for the local
gate at revision `d6e5ded` on 2026-09-05. Physical VRAM was 8,589,410,304 bytes on an
RTX 3070 under WDDM. Compression was disabled.

| Actual FP32 operand bytes | Approx. GiB | Retired tiles (two passes) | Maps = SetAccess = unmaps | Evictions / reuse |
| ---: | ---: | ---: | ---: | ---: |
| 9,448,338,752 | 8.799 | 1,122 | 284 | 187 |
| 12,884,112,128 | 11.999 | 1,530 | 386 | 289 |
| 17,178,816,512 | 15.999 | 2,040 | 512 | 415 |

The recorded CLOCK/LRU runs passed `abs(error) <= 1e-4 + 2e-5 * abs(reference)`,
accounting and cleanup checks. Full original reports/traces are referenced as
**local** `artifacts/...` files; they were not present in the uploaded source ZIP.
This is a sourced summary, not reconstructed raw evidence or independent reproduction.

These are operand sizes across a tiled computation, not simultaneous physical VRAM
residency. They do not prove superiority over another offload library or manual tiling.

## Reproduce with this preview

Run `preview.py smoke`, then `preview.py oversubscribe`. The helper verifies the
package inventory, binds report version/source to the package, invokes the real
controller with an explicit app-local cuBLAS pair, and saves stdout/stderr, report,
trace, result hashes and an offline HTML summary. It reuses the established schema,
numerical/accounting and trace validators rather than inventing missing fields.

The large-data claim additionally requires **observed** operand bytes above physical
VRAM plus actual eviction and handle reuse. A requested size or label alone is not
proof. Partial output, nonzero child exit, wrong revision, empty work, incomplete
trace or cleanup fail. Skipped/no-GPU outcomes never become successful GPU results.

For a new public result, retain the release/source SHA, archive checksum and CI link;
`report.json`, `trace.jsonl`, `run.json`, stdout/stderr from one run; and hardware,
driver/library configuration plus material external load. A demonstration video
must show that same run. Label any time compression; do not splice success outputs
from different revisions. The HTML report is not a substitute for raw evidence.

No automatic upload occurs. Review local paths and identifiers before publishing.
Hashes verify integrity, not authorship of a machine's measurements. Without an
independent observer or signature, a manifest cannot authenticate a GPU run.

## Claims intentionally excluded

No unchanged llama.cpp execution, arbitrary PyTorch support, training, support for
all GPUs, VRAM-speed system RAM, fixed capacity multiplier or universal acceleration
is claimed. The highly compressible Phase 5 fixture is not a typical LLM-weight
compression ratio. Phase 4b's Llama-like forward test is not a ready-made chatbot.
Phase 6b remains research with unproven gates.
