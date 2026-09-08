import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jsonschema
from xvram import compat_audit_execution as execution
from xvram import compat_launch_census as census
from xvram import compat_launch_probe as probe
from test_compat_audit_binary_contract import _completed


class ExecutionEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.paths = {key: self.root / (key + ".json") for key in
                      ("probe_report", "probe_trace", "census_report", "census_trace", "binary_evidence")}
        self.binary = _completed()
        self.write("binary_evidence", self.binary)
        rows = [dict(kind="setup", call_id=0, hooks=3)]
        for call, name in ((1, "foo"), (2, "bar")):
            rows.append(dict(kind="begin", call_id=call, api="cuLaunchKernel_resolved_legacy", metadata=True,
                             module_resolved=True, handle_kind="contextless_kernel", kernel_name=name, parameter_count=2))
            rows.extend(dict(kind="parameter", call_id=call, index=i, offset_bytes=offset, size_bytes=8)
                        for i, offset in enumerate((0, 16)))
            rows.append(dict(kind="end", call_id=call, result=0))
        self.lines("probe_trace", "xvram.cuda_launch_probe", rows)
        capture = dict(exit_code=0, timed_out=False, controller_reaped=True, process_tree_drained=True,
                       elapsed_ms=1.0, stdout_sha256="a"*64, output_truncated=False, errors=[])
        self.probe = probe.make_report("metadata", "driver_entry_probe",
                                      {"ggml-cuda.dll": self.binary["provenance"]["backend_sha256"]},
                                      capture, self.paths["probe_trace"], microbatch=128)
        self.write("probe_report", self.probe)
        rows = [dict(kind="session", terminal_complete=False, cupti_sha256=census.CUPTI_SHA256)]
        for call, name in ((1, "foo"), (2, "bar")):
            common = dict(api_id=call, parent_api_id=0, domain="driver", correlation_id=call,
                          probe_call_id=call, symbol="cuLaunchKernel")
            rows.extend([dict(kind="api_enter", **common), dict(kind="api_exit", result=0, **common),
                         dict(kind="kernel", correlation_id=call, name=name, start_ns=10, end_ns=20)])
        rows.append(dict(kind="summary", errors=0, dropped=0, buffers_outstanding=0, terminal_complete=False))
        self.lines("census_trace", "xvram.cuda_launch_census", rows)
        self.census = census.make_report(self.probe, self.paths["census_trace"])
        self.write("census_report", self.census)

    def write(self, key, value):
        self.paths[key].write_text(json.dumps(value), encoding="utf-8")

    def lines(self, key, record_type, rows):
        self.paths[key].write_text("".join(json.dumps(dict(schema_version=2, record_type=record_type,
            sequence=i, **row)) + "\n" for i, row in enumerate(rows, 1)), encoding="utf-8")

    def analyze(self):
        return execution.analyze(**self.paths)

    def test_exact_layout_candidates_do_not_become_bindings(self):
        report = self.analyze()
        self.assertEqual({r["name"]: r["status"] for r in report["kernels"]},
                         {"foo": "matching_candidate", "bar": "ambiguous_candidates"})
        self.assertTrue(report["coverage"]["observed_launches_reconciled"])
        self.assertFalse(any(g["proven"] for g in report["gates"].values()))
        execution.validate_report(report)
        schema = Path(__file__).resolve().parents[2] / "schemas/cuda-execution-evidence-v1.schema.json"
        jsonschema.validate(report, json.loads(schema.read_text()))

    def test_same_name_different_offset_size_order_or_count_is_mismatch(self):
        live = self.probe["trace"]["layouts"][:1]
        for changes in ([(0, 8)], [(0, 8), (8, 8)], [(0, 4), (16, 8)], [(16, 8), (0, 8)]):
            altered = copy.deepcopy(live)
            altered[0]["parameters"] = [dict(offset_bytes=o, size_bytes=s) for o, s in changes]
            rows = execution.reconcile_layouts(altered, self.binary["kernels"])
            self.assertEqual(rows[0]["status"], "abi_mismatch")
            self.assertFalse(rows[0]["binding_proven"])

    def test_missing_static_candidate_is_explicit(self):
        live = copy.deepcopy(self.probe["trace"]["layouts"][:1])
        live[0]["kernel_name"] = "unlisted"
        row = execution.reconcile_layouts(live, self.binary["kernels"])[0]
        self.assertEqual(row["status"], "no_static_candidate")

    def test_duplicate_names_and_raw_addresses_rejected(self):
        with self.assertRaises(ValueError):
            execution.reconcile_layouts([], self.binary["kernels"] * 2)
        live = copy.deepcopy(self.probe["trace"]["layouts"][:1])
        live[0]["kernel_name"] = "0x123456789abc"
        with self.assertRaises(ValueError):
            execution.reconcile_layouts(live, self.binary["kernels"])

    def test_other_backend_hash_cannot_be_joined(self):
        self.probe["binary_sha256"]["ggml-cuda.dll"] = "0" * 64
        self.write("probe_report", self.probe)
        with self.assertRaisesRegex(ValueError, "backend_provenance"):
            self.analyze()

    def test_modified_report_or_different_trace_rejected(self):
        self.census["observation"]["kernels"][0]["name"] = "other"
        self.write("census_report", self.census)
        with self.assertRaises(ValueError):
            self.analyze()

    def test_timeout_failed_input_not_analysis_success(self):
        self.probe["capture"].update(timed_out=True, exit_code=None)
        self.probe["exit_code"] = 26
        self.write("probe_report", self.probe)
        with self.assertRaises(ValueError):
            self.analyze()

    def test_trace_changed_during_analysis_rejected(self):
        with mock.patch.object(execution, "_hash", return_value="0"*64), self.assertRaisesRegex(ValueError, "input_changed"):
            self.analyze()

    def test_missing_footer_loss_or_truncation_rejected(self):
        original = self.paths["census_trace"].read_text()
        for text in (original.rsplit("\n", 2)[0] + "\n", original[:-8], original.replace('"dropped": 0', '"dropped": 1')):
            self.paths["census_trace"].write_text(text)
            with self.assertRaises(ValueError):
                self.analyze()

    def test_bounded_json_duplicate_fields_and_nonfinite(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '[]', 'null'):
            self.paths["probe_report"].write_text(value)
            with self.assertRaises(ValueError):
                execution._load(self.paths["probe_report"])
        self.paths["probe_report"].write_text(" " * 65)
        with mock.patch.object(execution, "CAP", 64), self.assertRaises(ValueError):
            execution._load(self.paths["probe_report"])

    def test_report_counter_claim_and_extra_fields_rejected(self):
        original = self.analyze()
        for edit in (lambda r: r["gates"]["cubin_binding"].update(proven=True),
                     lambda r: r["decision"].update(verdict="GO"),
                     lambda r: r["coverage"].update(metadata_calls=3),
                     lambda r: r["kernels"][0].update(binding_proven=True),
                     lambda r: r["kernels"][0].update(address=42),
                     lambda r: r["gates"]["memory_bounds"].update(blockers=[])):
            report = copy.deepcopy(original)
            edit(report)
            with self.assertRaises(ValueError):
                execution.validate_report(report)

    def test_cli_does_not_overwrite_and_only_serializes_normalized_evidence(self):
        args = []
        for key, path in self.paths.items():
            args.extend(["--" + key.replace("_", "-"), str(path)])
        output = self.root / "result.json"
        args.extend(["--json", str(output)])
        self.assertEqual(execution.main(args), 0)
        before = output.read_bytes()
        self.assertNotIn(str(self.root), before.decode())
        self.assertEqual(execution.main(args), 74)
        self.assertEqual(output.read_bytes(), before)

    def test_frozen_launch_v2_contracts(self):
        root = Path(__file__).resolve().parents[2] / "schemas"
        for name, expected in {
            "cuda-launch-census-trace-v2.schema.json": "82da749fadfdce78c809f49c9faaa92f0d7a7d3279da56c170d95d75527909b5",
            "cuda-launch-probe-trace-v2.schema.json": "74cbbef077e07aee1d26c0b0daa2ea6746f1ca9cfb2bf6aad0e5aaabc6ce2356",
        }.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
