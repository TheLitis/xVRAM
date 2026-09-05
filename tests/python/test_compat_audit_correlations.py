"""CPU-only record reconciliation, distinct from complete capture or GO."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import contextlib

import jsonschema
from xvram import compat_audit_correlations as correlation
from xvram import compat_audit_observations as observation
from test_compat_audit_analysis import session, complete, api


def launch(domain, correlation_id, **changes):
    fields = dict(domain=domain, symbol="cuLaunchKernel" if domain == "driver" else "cudaLaunchKernel",
                  context_id=2, stream_id=3, kernel_name="test_kernel", function_id=4,
                  grid_x=1, grid_y=1, grid_z=1, block_x=32, block_y=1, block_z=1, shared_bytes=0)
    fields.update(changes)
    return api(correlation_id, "launch", **fields)


def activity(kind="kernel", correlation_id=10, **changes):
    fields = dict(kind="activity", activity_kind=kind, correlation_id=correlation_id,
                  detail_known=True, start_ns=100, end_ns=200)
    if kind == "kernel":
        fields.update(context_id=2, stream_id=3, name="test_kernel", grid_x=1, grid_y=1, grid_z=1,
                      block_x=32, block_y=1, block_z=1, shared_bytes=0)
    fields.update(changes)
    return fields


def nested():
    outer, inner = launch("runtime", 10), launch("driver", 10)
    return [session(), outer[0], inner[0], inner[1], outer[1],
            activity(), activity("api_runtime")]


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "trace.jsonl"

    def write(self, records):
        values = []
        for sequence, record in enumerate(records, 1):
            values.append(dict(schema_version=1, report_type="xvram.cuda_compat_audit_trace",
                               sequence=sequence, timestamp_ns=sequence, **record))
        self.path.write_text("".join(json.dumps(record) + "\n" for record in values), encoding="utf-8")
        return self.path

    def result(self, records=None):
        return correlation.correlate_trace(self.write(complete(nested() if records is None else records)))

    def test_nested_layers_are_one_gpu_execution(self):
        result = self.result()
        self.assertTrue(result["launch_records_reconciled"])
        self.assertTrue(result["trace_complete"])
        self.assertEqual(result["gpu_kernel_activities"], 1)
        self.assertEqual(result["runtime_driver_gpu_chains"], 1)
        self.assertEqual(result["runtime_launch_callbacks"] + result["driver_launch_callbacks"], 2)
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["semantic_ranges_proven"])

    def test_direct_driver_and_deferred_activity(self):
        records = [session(), activity("api_driver"), activity(), *launch("driver", 10)]
        result = self.result(records)
        self.assertTrue(result["launch_records_reconciled"])
        self.assertEqual(result["direct_driver_launches"], 1)

    def test_incomplete_terminal_does_not_erase_observed_correlation(self):
        result = correlation.correlate_trace(self.write(nested()))
        self.assertTrue(result["launch_records_reconciled"])
        self.assertFalse(result["trace_complete"])
        self.assertFalse(result["execution_ready"])

    def test_unknown_or_changed_launch_fields_refused(self):
        for index, changes in ((3, {"shared_bytes": 32}), (2, {"kernel_name": "other"}),
                               (3, {"detail_known": False}), (3, {"block_x": 64}),
                               (3, {"context_id": 5}), (3, {"callback_id": 9}),
                               (5, {"detail_known": False}), (5, {"shared_bytes": 4}),
                               (6, {"start_ns": 0}), (6, {"detail_known": False})):
            with self.subTest(index=index, changes=changes):
                records = nested()
                records[index].update(changes)
                self.assertFalse(self.result(records)["launch_records_reconciled"])

    def test_cross_thread_correlation_collision(self):
        records = nested() + api(10, "other", domain="driver", thread_id=2)
        result = self.result(records)
        self.assertFalse(result["launch_records_reconciled"])
        self.assertIn("conflicting_callback_correlation", {item["code"] for item in result["issues"]})

    def test_duplicate_activity_and_wrong_stream(self):
        for record in (activity(), activity(stream_id=99)):
            self.assertFalse(self.result(nested() + [record])["launch_records_reconciled"])

    def test_failed_or_unclosed_runtime_is_not_a_valid_chain(self):
        records = nested()
        records[4]["status"] = 1
        self.assertFalse(self.result(records)["launch_records_reconciled"])
        del records[4]
        self.assertFalse(self.result(records)["launch_records_reconciled"])

    def test_out_of_order_api_stack_refused(self):
        records = nested()
        records[3], records[4] = records[4], records[3]
        self.assertFalse(self.result(records)["launch_records_reconciled"])

    def test_new_session_and_records_after_summary_refused(self):
        self.assertFalse(self.result(nested() + [session()])["launch_records_reconciled"])
        self.write(complete(nested()) + [activity()])
        self.assertFalse(correlation.correlate_trace(self.path)["launch_records_reconciled"])

    def test_malformed_suffix_does_not_reconcile_prefix(self):
        self.write(complete(nested()))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("{")
        self.assertFalse(correlation.correlate_trace(self.path)["launch_records_reconciled"])

    def test_changed_file_between_passes_refused(self):
        self.write(complete(nested()))
        reader = correlation._trace_records
        calls = 0
        def changing(path, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.path.write_text(self.path.read_text().replace('"end_ns": 200', '"end_ns": 201'))
            return reader(path, **kwargs)
        with mock.patch.object(correlation, "_trace_records", side_effect=changing):
            self.assertFalse(correlation.correlate_trace(self.path)["launch_records_reconciled"])

    def test_state_and_nesting_limits(self):
        with mock.patch.object(correlation, "MAX_STATE_ITEMS", 2):
            self.assertFalse(self.result()["launch_records_reconciled"])
        with mock.patch.object(correlation, "MAX_NESTING_DEPTH", 1):
            self.assertFalse(self.result()["launch_records_reconciled"])

    def test_wrapper_schema_hash_privacy_and_no_overwrite(self):
        self.write(complete(nested()))
        result = observation.observations([self.path])
        schema_path = Path(__file__).resolve().parents[2] / "schemas/cuda-compat-audit-observations-v1.schema.json"
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(result, schema)
        self.assertEqual(result["traces"][0]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertNotIn(str(self.path), json.dumps(result))
        self.assertEqual(result["decision"]["verdict"], "NO-GO")
        output = self.path.with_name("observations.json")
        self.assertEqual(observation.main(["--trace", str(self.path), "--json", str(output)]), 0)
        before = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(observation.main(["--trace", str(self.path), "--json", str(output)]), 74)
        self.assertEqual(before, output.read_bytes())

    def test_wrapper_refuses_invalid_trace_list(self):
        self.write(complete(nested()))
        for paths in ([], [self.path, self.path], [self.path] * 17):
            with self.assertRaises(ValueError):
                observation.observations(paths)


if __name__ == "__main__":
    unittest.main()
