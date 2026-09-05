"""No-driver tests for public resolver observations; no routing GO inference."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jsonschema

from xvram import compat_audit_analysis as audit
from xvram import compat_audit_correlations as correlations
from xvram import compat_audit_routing as routing


ROOT = Path(__file__).resolve().parents[2]


def resolver(correlation=1, api="cuGetProcAddress_v2", **kwargs):
    base = {"domain": "runtime" if api.startswith("cuda") else "driver", "callback_id": 1,
            "correlation_id": correlation, "thread_id": 1, "symbol": api, "op": "other",
            "detail_known": True, "requested_symbol": "cuLaunchKernel", "requested_version": 13030,
            "resolver_flags": 0}
    base.update(kwargs)
    return [dict(base, kind="api_enter"), dict(base, kind="api_exit", status=0, query_status=0, entry_point_id=1)]


class AuditRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads((ROOT / "schemas/cuda-compat-audit-trace-v2.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(cls.schema)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "trace.jsonl"

    def records(self, calls, version=2):
        session = {"kind": "session", "collector_version": "0.1", "process_id": 1,
                   "max_record_bytes": audit.MAX_RECORD_BYTES, "kernel_arguments_captured": False,
                   "tensor_bounds_known": False, "runtime_parameter_bytes_not_kernel_arguments": True,
                   "cublas_api_visibility": "unavailable", "timestamp_clock": "steady_clock", "activity_clock": "cupti"}
        records = [session, *calls]
        records.append({"kind": "summary", "callback_records": sum(r["kind"] in ("api_enter", "api_exit") for r in records),
                        "activity_records": sum(r["kind"] == "activity" for r in records),
                        "resource_records": sum(r["kind"] == "resource" for r in records),
                        "dropped_records": 0, "unknown_callback_details": 0, "unknown_activity_kinds": 0,
                        "serialization_errors": 0, "collector_errors": 0, "incomplete_activities": 0,
                        "buffers_requested": 0, "buffers_completed": 0, "complete": True,
                        "safely_finalized": True, "terminal_checkpoint": "explicit_finalize"})
        return [{"schema_version": version, "report_type": audit.TRACE_TYPE, "sequence": i,
                 "timestamp_ns": i, **r} for i, r in enumerate(records, 1)]

    def summarize(self, records):
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return routing.summarize_routing(self.path)

    def codes(self, result):
        return {issue["code"] for issue in result["issues"]}

    def test_opt_in_schema_and_v1_contract_are_frozen(self):
        expected = {"cuda-compat-audit-trace-v1.schema.json": "e7922d80d39b55c914c88739e5d97b84d117a3e67540b0507a6f1cebfef7a63f",
                    "cuda-compat-audit-v1.schema.json": "07b7d9284f3a091745b1bfc0126eac00cdcfd75bb0ee11496b5c17a94b351b5e"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / "schemas" / name).read_bytes()).hexdigest(), digest)
        for record in self.records(resolver()):
            jsonschema.validate(record, self.schema)
            audit.validate_trace_record(record)

    def test_success_observes_request_not_complete_routing(self):
        result = self.summarize(self.records(resolver()))
        self.assertEqual((result["resolver_calls"], result["resolved_calls"]), (1, 1))
        self.assertEqual(result["evidence"], "observed")
        self.assertFalse(result["routing_coverage_complete"])
        self.assertFalse(result["semantic_ranges_proven"])
        self.assertEqual(result["requests"][0]["requested_symbol"], "cuLaunchKernel")
        self.assertEqual(result["requests"][0]["entry_point_count"], 1)

    def test_nested_runtime_driver_are_separate_observations(self):
        runtime = resolver(1, "cudaGetDriverEntryPointByVersion")
        driver = resolver(2)
        result = self.summarize(self.records([runtime[0], *driver, runtime[1]]))
        self.assertEqual(result["resolver_calls"], 2)
        self.assertEqual(result["resolved_calls"], 2)
        self.assertEqual(len(result["requests"]), 2)

    def test_repeated_queries_are_allowed_but_not_identity_reuse(self):
        result = self.summarize(self.records([*resolver(1), *resolver(2)]))
        self.assertEqual(result["resolved_calls"], 2)
        self.assertEqual(result["requests"][0]["entry_point_count"], 1)
        self.assertNotIn("duplicate_resolver_identity", self.codes(result))
        result = self.summarize(self.records([*resolver(1), *resolver(1)]))
        self.assertIn("duplicate_resolver_identity", self.codes(result))
        self.assertEqual(result["evidence"], "unresolved")

    def test_failed_api_does_not_infer_output(self):
        calls = resolver()
        calls[-1].update(status=500)
        calls[-1].pop("query_status")
        calls[-1].pop("entry_point_id")
        result = self.summarize(self.records(calls))
        self.assertEqual(result["api_failed_calls"], 1)
        self.assertEqual(result["resolved_calls"], 0)

    def test_query_failure_and_null_or_unknown_result(self):
        for query in (1, 2, 99):
            calls = resolver()
            calls[-1]["query_status"] = query
            calls[-1].pop("entry_point_id")
            result = self.summarize(self.records(calls))
            self.assertEqual(result["query_failed_calls"], 1)
            self.assertEqual("unknown_resolver_query_status" in self.codes(result), query == 99)
        calls = resolver()
        calls[-1].pop("entry_point_id")
        result = self.summarize(self.records(calls))
        self.assertIn("successful_resolver_without_entry_point", self.codes(result))

    def test_old_public_resolver_has_no_query_status(self):
        calls = resolver(api="cuGetProcAddress")
        calls[-1].pop("query_status")
        result = self.summarize(self.records(calls))
        self.assertEqual(result["resolved_calls"], 1)
        calls = resolver(api="cudaGetDriverEntryPoint_ptsz")
        for record in calls:
            record.pop("requested_version")
        result = self.summarize(self.records(calls))
        self.assertIsNone(result["requests"][0]["requested_version"])

    def test_unknown_v1_metadata_is_not_reconstructed(self):
        calls = resolver()
        for record in calls:
            for field in ("requested_symbol", "requested_version", "resolver_flags", "query_status", "entry_point_id"):
                record.pop(field, None)
            record["detail_known"] = False
        result = self.summarize(self.records(calls, 1))
        self.assertEqual(result["unknown_calls"], 1)
        self.assertEqual(result["requests"], [])
        self.assertIn("resolver_metadata_unavailable", self.codes(result))

    def test_changed_inputs_and_unmatched_pairs_fail_closed(self):
        calls = resolver()
        calls[-1]["requested_version"] += 1
        self.assertIn("resolver_input_changed", self.codes(self.summarize(self.records(calls))))
        self.assertIn("unmatched_resolver_exit", self.codes(self.summarize(self.records([resolver()[-1]]))))
        self.assertIn("unclosed_resolver_calls", self.codes(self.summarize(self.records([resolver()[0]]))))

    def test_drop_terminal_and_sequence_cannot_manufacture_coverage(self):
        records = self.records(resolver())
        records[-1].update(complete=False, safely_finalized=False, terminal_checkpoint="process_exit_unflushed", dropped_records=1)
        records[1]["sequence"] = 10
        result = self.summarize(records)
        self.assertTrue({"collector_not_safely_finalized", "collector_incomplete_evidence", "sequence_gap"} <= self.codes(result))
        result = self.summarize(self.records(resolver())[:-1])
        self.assertIn("missing_summary", self.codes(result))

    def test_limits_and_malformed_input_are_redacted(self):
        with mock.patch.object(routing, "MAX_ROUTING_ITEMS", 0):
            self.assertIn("routing_state_limit", self.codes(self.summarize(self.records(resolver()))))
        with mock.patch.object(routing, "MAX_ROUTING_TEXT_BYTES", 0):
            self.assertIn("routing_text_limit", self.codes(self.summarize(self.records(resolver()))))
        self.path.write_text("broken private path C:/secret", encoding="utf-8")
        result = routing.summarize_routing(self.path)
        self.assertIn("invalid_routing_trace", self.codes(result))
        self.assertNotIn(str(self.path), json.dumps(result))
        self.assertNotIn("secret", json.dumps(result))

    def test_schema_rejects_addresses_wrong_domains_and_early_outputs(self):
        normal = self.records(resolver())[2]
        bad = [dict(normal, entry_point_id=0), dict(normal, query_status=1),
               dict(normal, requested_symbol="cuFoo_0x123abc"), dict(normal, requested_symbol="C:/private"),
               dict(normal, resolver_flags=True), dict(normal, requested_version=1 << 32),
               dict(normal, domain="runtime"), dict(normal, status=1), dict(normal, raw_pointer=1234),
               dict(normal, kind="api_enter"), dict(normal, symbol="cuMemAlloc_v2"),
               dict(normal, symbol="cuGetProcAddress"),
               dict(normal, domain="runtime", symbol="cudaGetDriverEntryPoint"),
               {key: value for key, value in normal.items() if key not in ("requested_symbol", "resolver_flags", "query_status", "entry_point_id", "requested_version")}]
        for record in bad:
            with self.subTest(record=record):
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(record, self.schema)
                with self.assertRaises(audit.AuditInputError):
                    audit.validate_trace_record(record)

    def test_all_existing_record_shapes_are_preserved_except_version(self):
        old = json.loads((ROOT / "schemas/cuda-compat-audit-trace-v1.schema.json").read_text())
        new = copy.deepcopy(self.schema)
        for before, after in zip(old["oneOf"], new["oneOf"], strict=True):
            after["properties"]["schema_version"]["const"] = 1
            if after["properties"]["kind"]["const"] in ("api_enter", "api_exit"):
                after.pop("allOf")
                for field in ("requested_symbol", "requested_version", "resolver_flags", "query_status", "entry_point_id"):
                    after["properties"].pop(field, None)
            self.assertEqual(before, after)

    def test_observations_schema_accepts_real_derived_shapes(self):
        schema = json.loads((ROOT / "schemas/cuda-compat-audit-observations-v1.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        result = self.summarize(self.records(resolver()))
        document = {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_observations",
                    "traces": [{"trace_id": 1, "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                                "input_unchanged": True, "routing": result,
                                "correlations": correlations.correlate_trace(self.path)}],
                    "decision": {"verdict": "NO-GO", "execution_ready": False, "oversubscription_proof": False},
                    "limitations": ["resolver_results_do_not_prove_invocation_routing"]}
        jsonschema.validate(document, schema)
        mutated = copy.deepcopy(document)
        mutated["traces"][0]["routing"]["requests"][0]["raw_pointer"] = 1234
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(mutated, schema)
        mutated = copy.deepcopy(document)
        mutated["decision"]["verdict"] = "GO"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(mutated, schema)
        mutated = copy.deepcopy(document)
        mutated["traces"][0]["routing"]["requests"][0]["requested_symbol"] = "C:/private/0x123abc"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(mutated, schema)
        mutated = copy.deepcopy(document)
        mutated["traces"][0]["correlations"]["device_ordering_proven"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(mutated, schema)


if __name__ == "__main__":
    unittest.main()
