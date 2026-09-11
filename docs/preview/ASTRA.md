# Development with GPT-6 Astra

## Attribution boundary

The maintainer, **TheLitis**, identifies commit
[`98e41accdf8b45b019354584fa49472dd40763f1`](https://github.com/TheLitis/xVRAM/commit/98e41accdf8b45b019354584fa49472dd40763f1)
— `build(compression): pin and stage lossless codecs` — as the first commit from
which development moved to GPT-6 Astra. This statement was supplied by the
maintainer on 2026-09-11. Earlier work used earlier GPT models.

This is a **maintainer-reported workflow boundary**. Git records code changes and
commit identities, not which model generated a suggestion. We do not claim that
Astra authored every line, independently designed the whole system, performed
physical GPU tests itself, or contributed to phases that predate this boundary.
The maintainer directs development, reviews decisions and runs local hardware tests.

The reviewed pre-preview snapshot was
[`39e2a3b012edaa4fad523ab72fa843dfc53942ad`](https://github.com/TheLitis/xVRAM/commit/39e2a3b012edaa4fad523ab72fa843dfc53942ad).
[The Git comparison](https://github.com/TheLitis/xVRAM/compare/98e41accdf8b45b019354584fa49472dd40763f1...39e2a3b012edaa4fad523ab72fa843dfc53942ad)
contains 68 subsequent commits. The boundary commit itself is not included in that
comparison. These counts are provenance, not a measure of quality or model autonomy.

## What changed after the transition

| Engineering area | Inspectable source / record | Boundary |
| --- | --- | --- |
| Lossless host backing | `src/residency/compression.cpp`, `cpu_codec_pool.cpp`, `nvcomp_pipeline.cpp`, `include/xvram/xvram_v2.h`; [Phase 5](../adaptive-compression.md) | Generation-safe representation changes, raw spill reservation, opt-in v2 paths. GPU results are maintainer-run, not independently reproduced in launch preparation. |
| Explicit CUDA/cuBLAS integration | `src/cuda_compat`, `src/compat_bench`; [Phase 6a](../cuda-compat.md) | Narrow synchronous FP32 profile for integrated applications, not arbitrary unchanged applications. |
| Evidence-led compatibility research | `src/compat_launch_probe`, `python/xvram/compat_audit_*`; [native witnesses](../cuda-execution-witness.md) | More identity, typed-argument and lifecycle evidence. All four profile-wide proof gates remain open. CPU-offload observations are not xVRAM oversubscription. |
| Developer preview | `examples/preview-gemm`, `scripts/preview.py`, `scripts/package_preview.py`, `tests/preview` | Public-API consumer, bounded run/validation path and CI-gated distribution. New GPU measurements require the packaged binaries on compatible hardware. |

An instructive research result was rejecting an apparently successful process exit
as insufficient proof of complete teardown: the diagnostic ledger observed 21 late
CUDA API calls after an earlier footer. The project documents that gap instead of
hiding it. See the [witness report](../cuda-execution-witness.md). No private
conversation transcript or fabricated prompt is presented as proof.

## Maker statement

> I started xVRAM with earlier GPT models and moved its development workflow to
> GPT-6 Astra beginning at commit 98e41acc. My goal was not another model wrapper:
> it was to use Astra while developing and reviewing a real C++20/CUDA systems
> codebase, including lossless memory backing, explicit integration and tests.
> I remain responsible for the project and for local GPU verification. This preview
> exposes a supported, reproducible slice rather than claiming the whole roadmap is
> complete. xVRAM does not call an OpenAI API at runtime.

This statement describes the maintainer's reported process. Specific prompt excerpts,
model session exports and before/after debugging stories can be attached separately
once attribution and privacy are reviewed; none is invented here.
