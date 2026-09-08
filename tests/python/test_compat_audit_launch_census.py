import copy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import jsonschema
from xvram import compat_launch_census as census


def fixture():
    rows = [dict(kind="session", terminal_complete=False, cupti_sha256=census.CUPTI_SHA256)]
    for cid, marker, domain in [(1, 1, "driver"), (2, 0, "runtime")]:
        args = dict(api_id=cid, parent_api_id=0, correlation_id=cid, probe_call_id=marker, domain=domain, symbol="cuLaunchKernel" if domain == "driver" else "cudaLaunchKernel")
        rows += [dict(kind="api_enter", **args), dict(kind="api_exit", result=0, **args)]
    rows += [dict(kind="kernel", correlation_id=i, name="test_kernel", start_ns=10, end_ns=20) for i in (1, 2, 3)]
    rows += [dict(kind="summary", terminal_complete=False, errors=0, dropped=0, buffers_outstanding=1)]
    return [dict(schema_version=1, record_type="xvram.cuda_launch_census", sequence=i, **r) for i, r in enumerate(rows, 1)]


class CensusTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "trace.jsonl"

    def analyze(self, rows):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return census.analyze(self.path, 1)

    def test_same_trace_partition(self):
        result = self.analyze(fixture())
        counts = result["counts"]
        self.assertEqual([counts[k] for k in ("marked_kernels", "unmarked_kernels", "uncorrelated_kernels")], [1, 1, 1])
        self.assertEqual(counts["probe_calls_with_gpu_activity"], 1)
        self.assertEqual(counts["buffers_outstanding"], 1)

    def test_buffered_activity_before_callback(self):
        rows = fixture()
        rows.insert(1, rows.pop(5))
        for i, row in enumerate(rows, 1):
            row["sequence"] = i
        self.assertEqual(self.analyze(rows)["counts"]["marked_kernels"], 1)

    def test_loss_and_missing_footer(self):
        for field in ("errors", "dropped"):
            rows = fixture()
            rows[-1][field] = 1
            with self.assertRaises(ValueError):
                self.analyze(rows)
        with self.assertRaises(ValueError):
            self.analyze(fixture()[:-1])

    def test_bad_fields_types_privacy_and_marker(self):
        changes = [(1, "probe_call_id", 2), (1, "correlation_id", True), (1, "correlation_id", 2**32),
                   (1, "symbol", "0x123456789abc"), (1, "address", 123), (1, "sequence", 8),
                   (5, "end_ns", 9), (0, "terminal_complete", True), (8, "terminal_complete", True),
                   (0, "cupti_sha256", "0" * 64)]
        for index, field, value in changes:
            with self.subTest(field=field, value=value):
                rows = fixture()
                rows[index][field] = value
                with self.assertRaises(ValueError):
                    self.analyze(rows)

    def test_marker_must_survive_until_exit(self):
        rows = fixture()
        rows[2]["probe_call_id"] = 0
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_open_api_not_counted_as_coverage(self):
        rows = fixture()
        del rows[2]
        for i, row in enumerate(rows, 1):
            row["sequence"] = i
        counts = self.analyze(rows)["counts"]
        self.assertEqual(counts["marked_kernels"], 0)
        self.assertEqual(counts["open_api_calls"], 1)

    def test_api_identity_reuse(self):
        rows = fixture()
        rows[3]["api_id"] = 1
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_shared_runtime_driver_correlation(self):
        rows = fixture()
        # A non-launch Driver helper may share a Runtime call's correlation ID.
        rows[1]["symbol"] = rows[2]["symbol"] = "cuCtxGetCurrent"
        rows[1]["probe_call_id"] = rows[2]["probe_call_id"] = 0
        rows[3]["correlation_id"] = rows[4]["correlation_id"] = 1
        counts = self.analyze(rows)["counts"]
        self.assertEqual(counts["marked_kernels"], 0)
        self.assertEqual(counts["unmarked_kernels"], 1)

    def test_parent_must_retire_after_child(self):
        rows = fixture()
        rows[3]["parent_api_id"] = 1  # parent already retired
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_nested_runtime_ancestor_is_not_a_conflicting_marker(self):
        rows = fixture()
        rows[3]["correlation_id"] = rows[4]["correlation_id"] = 1
        rows[1]["api_id"] = rows[2]["api_id"] = 2
        rows[1]["parent_api_id"] = rows[2]["parent_api_id"] = 1
        rows[3]["api_id"] = rows[4]["api_id"] = 1
        rows = [rows[0], rows[3], rows[1], rows[2], rows[4], *rows[5:]]
        for i, row in enumerate(rows, 1):
            row["sequence"] = i
        self.assertEqual(self.analyze(rows)["counts"]["marked_kernels"], 1)

    def test_sibling_marked_and_unmarked_launches_are_ambiguous(self):
        rows = fixture()
        rows[3]["correlation_id"] = rows[4]["correlation_id"] = 1
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_failed_launch_cannot_claim_gpu_activity(self):
        rows = fixture()
        rows[2]["result"] = 1
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_json_truncation_duplicate_and_limit(self):
        for raw in ('{"schema_version":1,"schema_version":1}\n', '{}', 'x' * 65537):
            self.path.write_text(raw)
            with self.assertRaises(ValueError):
                census.analyze(self.path, 1)

    def test_strict_contracts_and_false_proof(self):
        root = Path(__file__).resolve().parents[2] / "schemas"
        trace_schema = json.loads((root / "cuda-launch-census-trace-v1.schema.json").read_text())
        for row in fixture():
            jsonschema.validate(row, trace_schema)
        report = dict(schema_version=1, report_type="xvram.cuda_launch_census", version="0.1.0-dev",
                      cupti_sha256=census.CUPTI_SHA256,
                      observation=self.analyze(fixture()), diagnostics=[], proof=dict.fromkeys(census.PROOF, False),
                      exit_code=0, probe_trace_sha256="1" * 64)
        schema = json.loads((root / "cuda-launch-census-report-v1.schema.json").read_text())
        jsonschema.validate(report, schema)
        census.validate_report(report)
        for key in census.PROOF:
            bad = copy.deepcopy(report)
            bad["proof"][key] = True
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(bad, schema)
            with self.assertRaises(ValueError):
                census.validate_report(bad)
        report["observation"]["counts"]["marked_kernels"] += 1
        with self.assertRaises(ValueError):
            census.validate_report(report)

    def test_previous_launch_contracts_frozen(self):
        root = Path(__file__).resolve().parents[2] / "schemas"
        hashes = {"cuda-launch-probe-report-v1.schema.json": "fe3ca62df95d715defe4328a3f40afe5a7143912da3fc48c1687c7e6b5183efe",
                  "cuda-launch-probe-trace-v1.schema.json": "13d059cdb9a6fa577cfa4977eceba8d80608bce550046251a18890b0483abca7"}
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def test_previous_census_contracts_frozen(self):
        root = Path(__file__).resolve().parents[2] / "schemas"
        hashes = {"cuda-launch-census-report-v1.schema.json": "4f98c9d213b8c50b27bb6d7b74be0593532e699ac1c07dc7d941a79ceeef025b",
                  "cuda-launch-census-trace-v1.schema.json": "6857eacde169778cc1fa3958150c232940e1726ed3c6d2dd07b30a72aa209665"}
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)

    def test_trace_v2_has_separate_capacity_and_cannot_mix_versions(self):
        rows = fixture()
        self.analyze(rows)
        with mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=census.CAP + 1)):
            with self.assertRaises(ValueError):
                census.analyze(self.path, 1)
        for row in rows:
            row["schema_version"] = 2
        self.analyze(rows)
        with mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=census.CAP + 1)):
            self.assertEqual(census.analyze(self.path, 1)["counts"]["marked_kernels"], 1)
        root = Path(__file__).resolve().parents[2] / "schemas"
        schema = json.loads((root / "cuda-launch-census-trace-v2.schema.json").read_text())
        for row in rows:
            jsonschema.validate(row, schema)
        rows[-1]["schema_version"] = 1
        with self.assertRaises(ValueError):
            self.analyze(rows)

    def test_capture_failure_and_timeout_are_not_hidden(self):
        from xvram import compat_launch_probe as probe
        probe_path = self.path.with_name("probe.jsonl")
        common = dict(schema_version=1, record_type="xvram.cuda_launch_probe")
        rows = [dict(**common, kind="setup", call_id=0, sequence=1, hooks=2),
                dict(**common, kind="begin", call_id=1, sequence=2, api="cuLaunchKernel", metadata=False,
                     module_resolved=False, kernel_name="", parameter_count=0),
                dict(**common, kind="end", call_id=1, sequence=3, result=0)]
        probe_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        capture = dict(exit_code=0, timed_out=False, controller_reaped=True, process_tree_drained=True,
                       elapsed_ms=1.0, stdout_sha256="0" * 64, output_truncated=False, errors=[])
        self.analyze(fixture())
        for changes, expected in [({}, 0), ({"exit_code": 1}, 27), ({"timed_out": True, "exit_code": None}, 26),
                                  ({"process_tree_drained": False}, 27), ({"output_truncated": True}, 27)]:
            source = probe.make_report("routing", "driver_entry_probe", {"xvram_driver_launch_probe.dll": "0" * 64},
                                       {**capture, **changes}, probe_path, microbatch=128)
            report = census.make_report(source, self.path)
            self.assertEqual(report["exit_code"], expected)
            schema_dir = Path(__file__).resolve().parents[2] / "schemas"
            jsonschema.validate(source, json.loads((schema_dir / "cuda-launch-probe-report-v1.schema.json").read_text()))
            schema = json.loads((schema_dir / "cuda-launch-census-report-v1.schema.json").read_text())
            jsonschema.validate(report, schema)
        self.path.write_text("truncated")
        self.assertEqual(census.make_report(source, self.path)["exit_code"], 27)
        source["exit_code"] = 26
        source["capture"].update(timed_out=True, exit_code=None)
        self.assertEqual(census.make_report(source, self.path)["exit_code"], 26)


if __name__ == "__main__":
    unittest.main()
