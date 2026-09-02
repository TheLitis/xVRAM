from __future__ import annotations

import json
import unittest
from pathlib import Path

import jsonschema

from xvram.torch_report import (
    ALLOWLIST_HASH,
    EXIT_COMPLETED,
    EXIT_TIMEOUT,
    empty_report,
    finalize_proof,
    success_semantics,
    validate_report_envelope,
    validate_trace_record,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "pytorch-inference-report-v1.schema.json").read_text(encoding="utf-8"))


def plan() -> dict[str, object]:
    return {
        "device": 0,
        "model": "llama2-like",
        "model_ratio": 1.5,
        "layers": 31,
        "batch": 1,
        "sequence": 32,
        "hidden": 4096,
        "intermediate": 11008,
        "heads": 32,
        "dtype": "float16",
        "policy": "clock",
        "cache_target_bytes": 7 * 1024**3,
        "chunk_size_bytes": 64 * 1024**2,
        "device_headroom_bytes": 512 * 1024**2,
        "scratch_cap_bytes": 512 * 1024**2,
        "prefetch_distance": 2,
        "sdpa_backend": "math",
        "seed": "0x585652414d503034",
    }


class TorchReportTests(unittest.TestCase):
    def test_timeout_report_is_schema_valid(self) -> None:
        report = empty_report(plan(), exit_code=EXIT_TIMEOUT, status="timeout", message="deadline")
        jsonschema.Draft202012Validator(SCHEMA).validate(report)

    def test_success_requires_all_proofs(self) -> None:
        report = empty_report(plan(), exit_code=EXIT_COMPLETED, status="completed", message="ok")
        report["device"]["total_vram_bytes"] = 8 * 1024**3
        report["model"].update(
            {
                "logical_bytes": 12 * 1024**3,
                "parameter_bytes": 11 * 1024**3,
                "activation_bytes": 1 * 1024**3,
            }
        )
        report["graph"].update(
            {
                "hash": "a" * 64,
                "backend_hash": "a" * 64,
                "allowlist_hash": ALLOWLIST_HASH,
                "node_count": 2,
                "region_count": 2,
            }
        )
        report["cache"].update(
            {
                "maps": 3,
                "set_access": 3,
                "unmaps": 3,
                "misses": 3,
                "h2d_bytes": 4096,
                "evictions": 1,
                "frame_reuses": 1,
            }
        )
        report["execution"].update(
            {
                "leases_acquired": 2,
                "leases_sealed": 2,
                "leases_retired": 2,
                "events_recorded": 2,
                "events_retired": 2,
                "live_views_peak": 3,
                "regions_completed": 2,
                "region_timings_ms": [1.0, 2.0],
                "trace_complete": True,
            }
        )
        report["verification"].update(
            {"reference_digest": "a" * 64, "output_digest": "a" * 64}
        )
        report["proof"]["stable_addresses"] = True
        report["cleanup"]["complete"] = True
        finalize_proof(report)
        valid, errors = success_semantics(report)
        self.assertTrue(valid, errors)
        jsonschema.Draft202012Validator(SCHEMA).validate(report)

        report["cache"]["resident_peak_bytes"] = report["cache"]["target_bytes"] + 1
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("cache residency does not reconcile with its live target", errors)
        report["cache"]["resident_peak_bytes"] = 0

        report["verification"]["output_digest"] = "not-a-sha256"
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("verification.output_digest is not lowercase SHA-256", errors)
        report["verification"]["output_digest"] = "a" * 64

        report["device"]["identifiers_included"] = True
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("device identifier privacy flag does not match configuration", errors)

    def test_diagnostics_forbid_success(self) -> None:
        report = empty_report(plan(), exit_code=EXIT_COMPLETED, status="completed", message="ok")
        for key in report["proof"]:
            report["proof"][key] = True
        report["diagnostics"].append({"stage": "x", "code": "y", "message": "z"})
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("diagnostics are not empty", errors)

    def test_controller_envelope_rejects_raw_addresses(self) -> None:
        report = empty_report(plan())
        report["cache"]["raw_va"] = 0x1234
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            validate_report_envelope(report)

    def test_nested_final_validation_rejects_wrong_type_without_crashing(self) -> None:
        report = empty_report(plan())
        report["execution"]["leases_acquired"] = {"unexpected": [None]}
        with self.assertRaisesRegex(ValueError, "leases_acquired must be an integer"):
            validate_report_envelope(report)

    def test_nested_final_validation_rejects_missing_field(self) -> None:
        report = empty_report(plan())
        del report["cleanup"]["context_released"]
        with self.assertRaisesRegex(ValueError, "invalid cleanup fields"):
            validate_report_envelope(report)

    def test_success_requires_identical_graph_hashes_and_canonical_allowlist(self) -> None:
        report = empty_report(plan(), exit_code=EXIT_COMPLETED, status="completed", message="ok")
        for key in report["proof"]:
            report["proof"][key] = True
        report["graph"].update(
            {
                "hash": "a" * 64,
                "backend_hash": "b" * 64,
                "allowlist_hash": ALLOWLIST_HASH,
                "node_count": 1,
                "region_count": 1,
            }
        )
        report["execution"].update(
            {
                "leases_acquired": 1,
                "leases_sealed": 1,
                "leases_retired": 1,
                "events_recorded": 1,
                "events_retired": 1,
                "live_views_peak": 1,
                "regions_completed": 1,
                "region_timings_ms": [0.1],
                "trace_complete": True,
            }
        )
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("graph hash and backend hash are not equal and non-empty", errors)

    def test_success_reconciles_regions_leases_events_and_views(self) -> None:
        report = empty_report(plan(), exit_code=EXIT_COMPLETED, status="completed", message="ok")
        for key in report["proof"]:
            report["proof"][key] = True
        report["graph"].update(
            {
                "hash": "a" * 64,
                "backend_hash": "a" * 64,
                "allowlist_hash": ALLOWLIST_HASH,
                "node_count": 1,
                "region_count": 1,
            }
        )
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("lease and tiled GEMM regions do not reconcile", errors)
        self.assertIn("events are zero or do not reconcile with execution boundaries", errors)
        self.assertIn("managed tensor views are zero or not fully released", errors)

    def test_success_semantics_handles_malformed_input(self) -> None:
        valid, errors = success_semantics({"schema_version": 1})
        self.assertFalse(valid)
        self.assertTrue(errors[0].startswith("invalid report envelope:"))

    def test_zero_counter_report_cannot_claim_success(self) -> None:
        report = empty_report(
            plan(), exit_code=EXIT_COMPLETED, status="completed", message="fabricated"
        )
        report["graph"].update(
            {
                "hash": "a" * 64,
                "backend_hash": "a" * 64,
                "node_count": 1,
                "region_count": 1,
            }
        )
        report["execution"].update(
            {
                "tiled_gemm_regions": 1,
                "tiled_gemm_tiles": 1,
                "events_recorded": 1,
                "events_retired": 1,
                "regions_completed": 1,
                "region_timings_ms": [0.1],
                "trace_complete": True,
            }
        )
        report["verification"].update(
            {"reference_digest": "a" * 64, "output_digest": "a" * 64}
        )
        report["proof"].update({name: True for name in report["proof"]})
        report["cleanup"]["complete"] = True
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("device VRAM was not observed", errors)
        self.assertIn("cache did not perform observable managed transfers", errors)
        self.assertIn("model byte accounting is empty or inconsistent", errors)

    def test_tiled_only_graph_reconciles_without_external_views(self) -> None:
        report = empty_report(
            plan(), exit_code=EXIT_COMPLETED, status="completed", message="ok"
        )
        report["device"]["total_vram_bytes"] = 8 * 1024**3
        report["model"].update(
            {"logical_bytes": 1152, "parameter_bytes": 1024, "activation_bytes": 128}
        )
        report["graph"].update(
            {
                "hash": "a" * 64,
                "backend_hash": "a" * 64,
                "node_count": 1,
                "region_count": 1,
            }
        )
        report["execution"].update(
            {
                "tiled_gemm_regions": 1,
                "tiled_gemm_tiles": 2,
                "events_recorded": 2,
                "events_retired": 2,
                "regions_completed": 1,
                "region_timings_ms": [1.0],
                "trace_complete": True,
            }
        )
        report["cache"].update(
            {"maps": 2, "set_access": 2, "unmaps": 2, "misses": 2, "h2d_bytes": 128}
        )
        report["verification"].update(
            {"reference_digest": "a" * 64, "output_digest": "a" * 64}
        )
        report["proof"]["stable_addresses"] = True
        report["cleanup"]["complete"] = True
        finalize_proof(report)
        valid, errors = success_semantics(report)
        self.assertTrue(valid, errors)

    def test_serialized_native_identities_are_rejected_even_in_strings(self) -> None:
        report = empty_report(plan())
        report["diagnostics"].append(
            {"stage": "cuda", "code": "leak", "message": "raw_va=0x1234"}
        )
        with self.assertRaisesRegex(ValueError, "serialized CUDA address"):
            validate_report_envelope(report)
        record = {
            "schema_version": 1,
            "record_type": "xvram.pytorch_trace",
            "sequence": 1,
            "monotonic_ns": 1,
            "kind": "lease",
            "region_id": 1,
            "allocation_id": None,
            "operation": "retired",
            "bytes": 0,
            "reason": "stream_handle: 0xabcd",
        }
        with self.assertRaisesRegex(ValueError, "serialized CUDA address"):
            validate_trace_record(record)


if __name__ == "__main__":
    unittest.main()
