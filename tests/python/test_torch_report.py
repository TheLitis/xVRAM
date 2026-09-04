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
from xvram.torch_runtime import _apply_benchmark_telemetry


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "pytorch-inference-report-v1.schema.json").read_text(encoding="utf-8"))
SCHEMA_V2 = json.loads(
    (ROOT / "schemas" / "pytorch-inference-report-v2.schema.json").read_text(
        encoding="utf-8"
    )
)
TRACE_SCHEMA_V2 = json.loads(
    (ROOT / "schemas" / "pytorch-inference-trace-record-v2.schema.json").read_text(
        encoding="utf-8"
    )
)


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


def compressed_plan() -> dict[str, object]:
    value = plan()
    value.update(
        {
            "compression": "adaptive",
            "compression_codec": "lz4",
            "state_pattern": "incompressible",
            "host_store_cap_bytes": 16 * 1024**3,
            "host_headroom_bytes": 4 * 1024**3,
            "compression_scratch_cap_bytes": 256 * 1024**2,
            "codec_slots": 2,
            "codec_workers": 2,
        }
    )
    return value


class TorchReportTests(unittest.TestCase):
    def test_timeout_report_is_schema_valid(self) -> None:
        report = empty_report(plan(), exit_code=EXIT_TIMEOUT, status="timeout", message="deadline")
        jsonschema.Draft202012Validator(SCHEMA).validate(report)

    def test_compression_selects_v2_while_off_remains_frozen_v1(self) -> None:
        off = empty_report(plan())
        self.assertEqual(off["schema_version"], 1)
        self.assertNotIn("compression", off)
        jsonschema.Draft202012Validator(SCHEMA).validate(off)

        compressed = empty_report(compressed_plan())
        self.assertEqual(compressed["schema_version"], 2)
        self.assertEqual(compressed["build"]["bridge_api_version"], 2)
        self.assertEqual(compressed["compression"]["policy"], "adaptive")
        self.assertEqual(compressed["compression"]["codec"], "lz4")
        validate_report_envelope(compressed)
        jsonschema.Draft202012Validator(SCHEMA_V2).validate(compressed)

    def test_compression_mapper_exposes_effective_budget_scratch_and_codec_events(
        self,
    ) -> None:
        report = empty_report(compressed_plan())
        telemetry = {
            "events_recorded": 2,
            "events_retired": 2,
            "gemm_events_recorded": 3,
            "gemm_events_retired": 3,
            "codec_events_recorded": 5,
            "codec_events_retired": 5,
            "effective_host_store_cap_bytes": 12 << 30,
            "effective_host_headroom_bytes": 3 << 30,
            "host_budget_bytes": 6 << 30,
            "host_budget_peak_bytes": 7 << 30,
            "conversion_scratch_peak_bytes": 96 << 20,
            "d2h_bytes": 12 << 20,
            "logical_d2h_bytes": 12 << 20,
            "pcie_d2h_bytes": 8 << 20,
            "pcie_d2h_payload_bytes": (8 << 20) - 4096,
            "pcie_d2h_metadata_bytes": 4096,
            "rejected_candidate_logical_d2h_bytes": 0,
            "hot_allocation_d2h_bytes": 5 << 20,
            "non_hot_allocation_d2h_bytes": 7 << 20,
            "logical_bytes": 9 << 30,
            "host_stored_bytes": 5 << 30,
            "host_stored_peak_bytes": 6 << 30,
            "host_raw_bytes": 2 << 30,
            "host_compressed_bytes": 3 << 30,
        }

        _apply_benchmark_telemetry(
            report,
            telemetry,
            close_succeeded=True,
            persistent_read_only=True,
        )

        self.assertEqual(report["execution"]["events_recorded"], 10)
        self.assertEqual(report["execution"]["events_retired"], 10)
        self.assertEqual(report["compression"]["host_store_cap_bytes"], 12 << 30)
        self.assertEqual(report["compression"]["host_headroom_bytes"], 3 << 30)
        self.assertEqual(report["compression"]["host_budget_bytes"], 6 << 30)
        self.assertEqual(report["compression"]["host_budget_peak_bytes"], 7 << 30)
        self.assertEqual(
            report["compression"]["conversion_scratch_peak_bytes"], 96 << 20
        )
        self.assertEqual(report["compression"]["codec_events_recorded"], 5)
        self.assertEqual(report["compression"]["codec_events_retired"], 5)
        self.assertEqual(report["cache"]["weight_d2h_bytes"], 5 << 20)
        self.assertEqual(
            report["compression"]["non_hot_allocation_d2h_bytes"], 7 << 20
        )
        self.assertEqual(
            report["compression"]["pcie_d2h_payload_bytes"],
            (8 << 20) - 4096,
        )
        self.assertEqual(report["compression"]["pcie_d2h_metadata_bytes"], 4096)

    def test_compression_v2_semantics_reconcile_storage_and_transfers(self) -> None:
        report = empty_report(
            compressed_plan(),
            exit_code=EXIT_COMPLETED,
            status="completed",
            message="ok",
        )
        gib = 1024**3
        report["device"]["total_vram_bytes"] = 8 * gib
        report["model"].update(
            {
                "logical_bytes": 12 * gib,
                "parameter_bytes": 11 * gib,
                "activation_bytes": 1 * gib,
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
                "d2h_bytes": 8192,
                "evictions": 1,
                "frame_reuses": 1,
            }
        )
        report["execution"].update(
            {
                "leases_acquired": 2,
                "leases_sealed": 2,
                "leases_retired": 2,
                "events_recorded": 4,
                "events_retired": 4,
                "live_views_peak": 3,
                "regions_completed": 2,
                "region_timings_ms": [1.0, 2.0],
                "trace_complete": True,
            }
        )
        report["verification"].update(
            {"reference_digest": "a" * 64, "output_digest": "a" * 64}
        )
        report["compression"].update(
            {
                "logical_bytes": 12 * gib,
                "stored_bytes": 5 * gib,
                "stored_peak_bytes": 6 * gib,
                "raw_bytes": 1 * gib,
                "compressed_bytes": 4 * gib,
                "host_budget_bytes": 7 * gib,
                "host_budget_peak_bytes": 8 * gib,
                "conversion_scratch_peak_bytes": 256 * 1024**2,
                "logical_h2d_bytes": 4096,
                "pcie_h2d_bytes": 4608,
                "pcie_h2d_payload_bytes": 4096,
                "pcie_h2d_metadata_bytes": 512,
                "logical_d2h_bytes": 12288,
                "pcie_d2h_bytes": 12800,
                "pcie_d2h_payload_bytes": 12288,
                "pcie_d2h_metadata_bytes": 512,
                "rejected_candidate_logical_d2h_bytes": 4096,
                "hot_allocation_d2h_bytes": 0,
                "non_hot_allocation_d2h_bytes": 8192,
                "raw_path_decisions": 1,
                "compression_attempts": 1,
                "compression_commits": 1,
                "generations_created": 1,
                "generations_committed": 1,
                "codec_events_recorded": 2,
                "codec_events_retired": 2,
            }
        )
        report["proof"]["stable_addresses"] = True
        report["cleanup"]["complete"] = True
        finalize_proof(report)
        valid, errors = success_semantics(report)
        self.assertTrue(valid, errors)
        validate_report_envelope(report)
        jsonschema.Draft202012Validator(SCHEMA_V2).validate(report)

        # Codec metadata is physical transport, not logical payload: totals may
        # exceed logical bytes when the split and payload bounds reconcile.
        self.assertGreater(
            report["compression"]["pcie_h2d_bytes"],
            report["compression"]["logical_h2d_bytes"],
        )
        self.assertGreater(
            report["compression"]["pcie_d2h_bytes"],
            report["compression"]["logical_d2h_bytes"],
        )

        report["compression"]["pcie_h2d_bytes"] = 4609
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn(
            "H2D PCIe bytes do not reconcile with payload and metadata", errors
        )

        report["compression"]["pcie_h2d_bytes"] = 4608
        report["compression"]["pcie_h2d_payload_bytes"] = 4097
        report["compression"]["pcie_h2d_metadata_bytes"] = 511
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("H2D PCIe payload exceeds logical H2D bytes", errors)

        report["compression"]["pcie_h2d_payload_bytes"] = 4096
        report["compression"]["pcie_h2d_metadata_bytes"] = 512
        report["compression"]["generations_discarded"] = 1
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("compression generations do not reconcile", errors)

        report["compression"]["generations_discarded"] = 0
        report["compression"]["host_budget_peak_bytes"] = (
            report["compression"]["host_store_cap_bytes"] + 1
        )
        valid, errors = success_semantics(report)
        self.assertFalse(valid)
        self.assertIn("compression host budget exceeds its effective cap", errors)

    def test_compression_v2_semantics_reject_unreconciled_native_counters(self) -> None:
        report = empty_report(
            compressed_plan(),
            exit_code=EXIT_COMPLETED,
            status="completed",
            message="ok",
        )
        gib = 1024**3
        report["device"]["total_vram_bytes"] = 8 * gib
        report["model"].update(
            logical_bytes=12 * gib,
            parameter_bytes=11 * gib,
            activation_bytes=1 * gib,
        )
        report["graph"].update(
            hash="a" * 64,
            backend_hash="a" * 64,
            allowlist_hash=ALLOWLIST_HASH,
            node_count=1,
            region_count=1,
        )
        report["execution"].update(
            leases_acquired=1,
            leases_sealed=1,
            leases_retired=1,
            events_recorded=2,
            events_retired=2,
            live_views_peak=1,
            regions_completed=1,
            region_timings_ms=[1.0],
            trace_complete=True,
        )
        report["cache"].update(
            maps=1,
            set_access=1,
            unmaps=1,
            misses=1,
            h2d_bytes=4096,
        )
        report["verification"].update(
            reference_digest="a" * 64,
            output_digest="a" * 64,
        )
        report["compression"].update(
            logical_bytes=12 * gib,
            stored_bytes=5 * gib,
            stored_peak_bytes=6 * gib,
            raw_bytes=1 * gib,
            compressed_bytes=4 * gib,
            host_budget_bytes=7 * gib,
            host_budget_peak_bytes=8 * gib,
            conversion_scratch_peak_bytes=256 * 1024**2,
            logical_h2d_bytes=4096,
            pcie_h2d_bytes=2048,
            pcie_h2d_payload_bytes=2048,
            pcie_h2d_metadata_bytes=0,
            raw_path_decisions=1,
            generations_created=1,
            generations_committed=1,
            codec_events_recorded=1,
            codec_events_retired=1,
        )
        report["proof"]["stable_addresses"] = True
        report["cleanup"]["complete"] = True
        finalize_proof(report)

        mutations = (
            (
                "logical_bytes",
                12 * gib - 1,
                "compression logical bytes do not reconcile with the model",
            ),
            (
                "logical_h2d_bytes",
                4095,
                "compression logical H2D bytes do not reconcile with the cache",
            ),
            (
                "logical_d2h_bytes",
                1,
                "compression logical D2H bytes do not reconcile with the cache",
            ),
            (
                "pcie_h2d_metadata_bytes",
                1,
                "H2D PCIe bytes do not reconcile with payload and metadata",
            ),
            (
                "rejected_candidate_logical_d2h_bytes",
                1,
                "compression logical D2H bytes do not reconcile with the cache",
            ),
            (
                "codec_events_retired",
                0,
                "compression codec events do not reconcile",
            ),
            (
                "hot_allocation_d2h_bytes",
                1,
                "measured weight D2H bytes do not reconcile with the cache",
            ),
            (
                "non_hot_allocation_d2h_bytes",
                1,
                "per-allocation-class D2H bytes do not reconcile",
            ),
            (
                "host_budget_bytes",
                4 * gib,
                "compression backing bytes exceed the charged host budget",
            ),
        )
        for field, invalid, expected in mutations:
            with self.subTest(field=field):
                original = report["compression"][field]
                report["compression"][field] = invalid
                valid, errors = success_semantics(report)
                self.assertFalse(valid)
                self.assertIn(expected, errors)
                report["compression"][field] = original

    def test_compression_trace_v2_has_generation_and_codec_identity(self) -> None:
        record = {
            "schema_version": 2,
            "record_type": "xvram.pytorch_trace",
            "sequence": 1,
            "monotonic_ns": 1,
            "kind": "codec",
            "region_id": 1,
            "allocation_id": 7,
            "operation": "decode_retired",
            "bytes": 4096,
            "reason": "adaptive_cost_win",
            "chunk_index": 2,
            "source_generation": 3,
            "target_generation": 4,
            "slot_generation": 4,
            "representation": "lz4_blocks",
            "codec_path": "cpu_lz4_gpu_decode",
        }
        validate_trace_record(record)
        record["reason"] = "device_pointer=0x1234"
        with self.assertRaisesRegex(ValueError, "serialized CUDA address"):
            validate_trace_record(record)
        jsonschema.Draft202012Validator(TRACE_SCHEMA_V2).validate(record)

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
        for identity in (
            "raw_va=0x1234",
            "device_pointer=0x12345678",
            "failed at 0x00007ffdeadc0de0",
        ):
            with self.subTest(identity=identity):
                report = empty_report(plan())
                report["diagnostics"].append(
                    {"stage": "cuda", "code": "leak", "message": identity}
                )
                with self.assertRaisesRegex(ValueError, "serialized CUDA address"):
                    validate_report_envelope(report)

        for identity in (
            "stream_handle: 0xabcd",
            "host_pointer: 0x12345678",
            "failed at 0x00007ffdeadc0de0",
        ):
            with self.subTest(identity=identity):
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
                    "reason": identity,
                }
                with self.assertRaisesRegex(ValueError, "serialized CUDA address"):
                    validate_trace_record(record)


if __name__ == "__main__":
    unittest.main()
