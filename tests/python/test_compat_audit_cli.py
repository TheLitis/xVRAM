from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jsonschema

from xvram import compat_audit


class AuditCliTests(unittest.TestCase):
    def validate(self, value):
        schema = Path(__file__).resolve().parents[2] / "schemas/cuda-compat-audit-v1.schema.json"
        jsonschema.Draft202012Validator(json.loads(schema.read_text())).validate(value)

    def test_inventory_without_driver_is_completed_not_go(self):
        with tempfile.TemporaryDirectory() as raw:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = compat_audit.main(["--output-dir", raw, "--json", "-", "--no-text"])
            report = json.loads(output.getvalue())
            self.validate(report)
            self.assertEqual(status, 0)
            self.assertEqual(report["decision"]["verdict"], "NO-GO")
            self.assertIsNone(report["configuration"]["cache_budget_bytes"])
            self.assertFalse(report["decision"]["interoperability_impossible_proven"])

    def test_corrupt_trace_still_emits_strict_report(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            trace = directory / "broken.jsonl"
            trace.write_text('{"schema_version":1')
            status = compat_audit.main(["--stage", "analyze", "--input-trace", str(trace),
                                       "--output-dir", raw, "--no-text"])
            report = json.loads((directory / "report.json").read_text())
            self.validate(report)
            self.assertEqual(status, 64)
            self.assertFalse(report["decision"]["execution_ready"])

    def test_capture_refuses_existing_output_without_overwriting(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            sentinel = directory / "capture.json"
            sentinel.write_text("existing")
            with contextlib.redirect_stderr(io.StringIO()):
                status = compat_audit.main(["--stage", "capture", "--binary-dir", raw,
                                           "--model-dir", raw, "--output-dir", raw])
            self.assertEqual(status, 23)
            self.assertEqual(sentinel.read_text(), "existing")

    def _fake_capture(self, result):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            metadata = {"version": 3, "file_size_bytes": 32, "metadata_sha256": "a" * 64,
                        "metadata_bytes_read": 32, "tensor_count": 0, "metadata_count": 0,
                        "known_tensor_bytes": 0, "unknown_tensor_types": [], "evidence": "unresolved",
                        "tensors": [], "architecture": None, "alignment": 32, "layout_source": None}
            with (mock.patch.object(compat_audit, "verify_binary_directory", return_value=[]),
                  mock.patch.object(compat_audit, "verify_model", return_value=[directory / "model.gguf"]),
                  mock.patch("xvram.compat_audit_analysis.read_gguf_metadata", return_value=metadata),
                  mock.patch.object(compat_audit, "capture_platform_supported", return_value=True),
                  mock.patch.object(compat_audit, "_gpu_sample", return_value={"available": False}),
                  mock.patch.object(compat_audit, "run_process", return_value=result)):
                status = compat_audit.main(["--stage", "all", "--capture-mode", "baseline",
                                           "--binary-dir", raw, "--model-dir", raw,
                                           "--output-dir", str(directory / "run"), "--no-text"])
            report = json.loads((directory / "run/report.json").read_text())
            self.validate(report)
            return status, report

    def test_timeout_report_never_claims_cleanup_or_go(self):
        status, report = self._fake_capture({"exit_code": 26, "timed_out": True,
                                            "controller_reaped": True, "process_tree_drained": True,
                                            "errors": ["deadline_exceeded"]})
        self.assertEqual(status, 26)
        self.assertEqual(report["decision"]["verdict"], "NO-GO")
        self.assertTrue(report["cleanup"]["controller_reaped"])
        self.assertIsNone(report["cleanup"]["collector_finalized"])
        self.assertIsNone(report["cleanup"]["complete"])
        self.assertIsNotNone(report["provenance"]["observation_file_sha256"])

    def test_native_failure_is_not_completed_audit(self):
        status, report = self._fake_capture({"exit_code": 17, "timed_out": False,
                                            "controller_reaped": True, "errors": []})
        self.assertEqual(status, 27)
        self.assertEqual(report["outcome"]["status"], "failed")
        self.assertFalse(report["decision"]["execution_ready"])


if __name__ == "__main__":
    unittest.main()
