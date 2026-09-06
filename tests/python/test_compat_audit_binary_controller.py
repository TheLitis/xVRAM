"""No-driver tests for the contained static-binary inspection controller."""

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

from xvram import compat_audit_binary as binary
from xvram import compat_audit_binary_contract as contract
from xvram import compat_audit_capture as capture


def _completed_worker():
    report = binary._empty_report()
    report["provenance"].update(
        backend_sha256=contract.BACKEND_SHA256, tool_sha256=contract.TOOL_SHA256,
        tool_version=contract.TOOL_VERSION,
        reports=[{"sha256": "a" * 64, "kernel_activities": 1}])
    report["modules"] = [{"module_index": 1, "sha256": "b" * 64, "size_bytes": 256,
                           "elf_flags": 0, "symbol_functions": 1, "selected_functions": 1,
                           "parameter_dump_sha256": "c" * 64}]
    report["kernels"] = [{
        "name": "test_kernel", "activities": 1, "status": "static_layout_only",
        "candidates": [{"module_index": 1, "cubin_sha256": "b" * 64,
                        "symbol_index": 1, "section_index": 1, "code_bytes": 32,
                        "parameter_bytes": 8,
                        "parameters": [{"ordinal": 0, "offset_bytes": 0, "size_bytes": 8}]}],
        "runtime_binding_proven": False, "memory_bounds_proven": False}]
    report["coverage"].update(sm86_cubins=1, candidate_modules=1, observed_types=1,
                              static_layout_types=1)
    report["tools"].update(invocations=4, stdout_bytes=128,
                           all_direct_children_reaped=True, all_pipes_drained=True)
    report["outcome"].update(status="completed", exit_code=0)
    contract.validate_report(report, allow_pending_cleanup=True)
    return report


class BinaryControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="xvram-binary-controller-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.backend = self.inputs / "ggml-cuda.dll"
        self.tool = self.inputs / "cuobjdump.exe"
        self.observation = self.inputs / "observation.json"
        for path in (self.backend, self.tool, self.observation):
            path.write_bytes(b"untrusted input; tests never execute this file")
        self.neighbor = self.work / "keep-existing-data.txt"
        self.neighbor.write_bytes(b"must remain unchanged")
        self.scratch_paths = []
        self.worker_plans = []

    def arguments(self, destination="-", *, timeout=7):
        return ["--backend", str(self.backend), "--cuobjdump", str(self.tool),
                "--report", str(self.observation), "--work-root", str(self.work),
                "--timeout-seconds", str(timeout), "--json", str(destination)]

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = binary.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def decoded(self, code, text, expected):
        report = json.loads(text)
        self.assertEqual(code, expected)
        self.assertEqual(report["outcome"]["exit_code"], expected)
        contract.validate_report(report)
        self.assertEqual(report["decision"], {
            "verdict": "NO-GO", "execution_ready": False, "oversubscription_proof": False})
        self.assertNotIn(str(self.root), text)
        self.assertEqual(self.neighbor.read_bytes(), b"must remain unchanged")
        return report

    def fake_capture(self, payload, *, exit_code=0, **updates):
        def run(command, **kwargs):
            self.assertEqual(command[:4], [sys.executable, "-m", "xvram.compat_audit_binary", "--worker"])
            self.assertEqual(len(command), 5)
            plan_path = Path(command[4])
            scratch = plan_path.parent
            self.scratch_paths.append(scratch)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.worker_plans.append(plan)
            self.assertEqual(scratch.parent, self.work)
            self.assertTrue(scratch.name.startswith("xvram-binary-audit-"))
            self.assertEqual(kwargs["cwd"], scratch)
            self.assertEqual(kwargs["output_dir"], scratch / "control")
            self.assertEqual(kwargs["timeout_seconds"], 7)
            self.assertEqual(Path(plan["scratch"]), scratch)
            self.assertEqual(Path(plan["backend"]), self.backend)
            self.assertEqual(Path(plan["tool"]), self.tool)
            self.assertEqual(plan["reports"], [str(self.observation)])
            (scratch / "owned-marker.bin").write_bytes(b"owned extraction marker")
            if payload is not None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                (scratch / "result.json").write_bytes(data)
            result = {"exit_code": exit_code, "timed_out": False, "errors": [],
                      "controller_reaped": True, "process_tree_drained": True,
                      "output_truncated": False}
            result.update(updates)
            return result
        return run

    def fake_run(self, payload, *, exit_code=0, **updates):
        with mock.patch.object(capture, "run_process", side_effect=self.fake_capture(
                payload, exit_code=exit_code, **updates)) as process:
            result = self.invoke(self.arguments())
        self.assertEqual(process.call_count, 1)
        return result

    def test_completed_worker_requires_controller_cleanup_and_stays_no_go(self):
        with mock.patch.object(binary, "_file_hash", side_effect=AssertionError("parent read input bytes")):
            code, text, stderr = self.fake_run(_completed_worker())
        report = self.decoded(code, text, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(report["outcome"]["status"], "completed")
        self.assertTrue(all(report["cleanup"].values()))
        self.assertEqual(report["coverage"]["static_layout_types"], 1)
        self.assertFalse(self.scratch_paths[0].exists())
        self.assertEqual(set(self.work.iterdir()), {self.neighbor})

    def test_valid_rejected_and_failed_worker_outcomes_are_preserved(self):
        for status, expected in (("rejected", 23), ("failed", 27), ("timeout", 26)):
            with self.subTest(status=status):
                payload = binary._empty_report()
                payload["outcome"].update(status=status, exit_code=expected)
                payload["diagnostics"] = ["worker_preflight_rejected"]
                contract.validate_report(payload, allow_pending_cleanup=True)
                code, text, _ = self.fake_run(payload, exit_code=expected)
                report = self.decoded(code, text, expected)
                self.assertTrue(all(report["cleanup"].values()))
                self.assertEqual(report["diagnostics"], ["worker_preflight_rejected"])

    def test_outer_timeout_ignores_partial_worker_json_and_cleans_confirmed_tree(self):
        code, text, _ = self.fake_run(b"{partial", exit_code=26, timed_out=True,
                                    errors=["deadline_exceeded"])
        report = self.decoded(code, text, 26)
        self.assertEqual(report["outcome"]["status"], "timeout")
        self.assertTrue(all(report["cleanup"].values()))
        self.assertIn("binary_inspection_timeout", report["diagnostics"])
        self.assertFalse(self.scratch_paths[0].exists())

    def test_crashed_worker_without_final_result_is_failed_not_rejected(self):
        code, text, _ = self.fake_run(None, exit_code=19)
        report = self.decoded(code, text, 27)
        self.assertEqual(report["outcome"]["status"], "failed")
        self.assertTrue(all(report["cleanup"].values()))

    def test_malformed_missing_and_oversized_worker_results_fail_closed(self):
        for payload in (None, b"{broken", b"[]", b"null", b"\xff", b'{"schema_version":1,"schema_version":1}',
                        b" " * (binary.MAX_EVIDENCE_BYTES + 1)):
            with self.subTest(payload_type=type(payload).__name__, length=len(payload) if payload else 0):
                code, text, _ = self.fake_run(payload)
                report = self.decoded(code, text, 27)
                self.assertEqual(report["outcome"]["status"], "failed")
                self.assertTrue(all(report["cleanup"].values()))

    def test_exit_mismatch_cannot_accept_a_valid_completed_worker_report(self):
        code, text, _ = self.fake_run(_completed_worker(), exit_code=23)
        report = self.decoded(code, text, 27)
        self.assertEqual(report["outcome"]["status"], "failed")
        self.assertTrue(all(report["cleanup"].values()))

    def test_real_validator_rejects_worker_schema_or_proof_spoof(self):
        for change in ("extra", "proof", "coverage", "partial_cleanup"):
            payload = _completed_worker()
            if change == "extra":
                payload["native_handle"] = "0x12345678"
            elif change == "proof":
                payload["decision"]["execution_ready"] = True
            elif change == "coverage":
                payload["coverage"]["observed_types"] = 2
            else:
                payload["cleanup"]["worker_reaped"] = True
            with self.subTest(change=change):
                code, text, _ = self.fake_run(payload)
                report = self.decoded(code, text, 27)
                self.assertEqual(report["outcome"]["status"], "failed")
                self.assertNotIn("0x12345678", text)
                self.assertTrue(all(report["cleanup"].values()))

    def test_unconfirmed_tree_retains_only_exact_owned_scratch(self):
        for reaped in (False, True):
            with self.subTest(reaped=reaped):
                with mock.patch.object(binary, "_remove_scratch", wraps=binary._remove_scratch) as remove:
                    code, text, _ = self.fake_run(_completed_worker(), controller_reaped=reaped,
                                                process_tree_drained=False, errors=["process_tree_not_drained"])
                report = self.decoded(code, text, 27)
                self.assertEqual(report["cleanup"], {"worker_reaped": reaped,
                    "process_tree_drained": False, "scratch_removed": False})
                remove.assert_not_called()
                scratch = self.scratch_paths[-1]
                self.assertEqual((scratch / "owned-marker.bin").read_bytes(), b"owned extraction marker")
                self.assertEqual(scratch.parent, self.work)

    def test_worker_cannot_spoof_scratch_removal_or_containment(self):
        payload = _completed_worker()
        payload["cleanup"] = dict.fromkeys(payload["cleanup"], True)
        contract.validate_report(payload, allow_pending_cleanup=True)
        with mock.patch.object(binary, "_remove_scratch", wraps=binary._remove_scratch) as remove:
            code, text, _ = self.fake_run(payload, controller_reaped=False, process_tree_drained=False)
        report = self.decoded(code, text, 27)
        self.assertFalse(any(report["cleanup"].values()))
        remove.assert_not_called()
        self.assertTrue((self.scratch_paths[0] / "owned-marker.bin").is_file())

    def test_cleanup_failure_is_failed_and_preserves_containment_facts(self):
        with mock.patch.object(binary, "_remove_scratch", side_effect=PermissionError("private cleanup path")) as remove:
            code, text, _ = self.fake_run(_completed_worker())
        report = self.decoded(code, text, 27)
        remove.assert_called_once_with(self.scratch_paths[0], self.work)
        self.assertEqual(report["cleanup"], {"worker_reaped": True,
            "process_tree_drained": True, "scratch_removed": False})
        self.assertIn("binary_scratch_cleanup_failed", report["diagnostics"])
        self.assertNotIn("private cleanup path", text)
        self.assertTrue(self.scratch_paths[0].is_dir())

    def test_capture_failure_before_cleanup_proof_retains_scratch(self):
        with mock.patch.object(capture, "run_process", side_effect=OSError("private capture path")), \
                mock.patch.object(binary, "_remove_scratch", wraps=binary._remove_scratch) as remove:
            code, text, _ = self.invoke(self.arguments())
        report = self.decoded(code, text, 27)
        self.assertFalse(any(report["cleanup"].values()))
        remove.assert_not_called()
        retained = [path for path in self.work.iterdir() if path.is_dir()]
        self.assertEqual(len(retained), 1)
        self.assertTrue((retained[0] / "plan.json").is_file())
        self.assertNotIn("private capture path", text)

    def test_capture_errors_or_truncation_cannot_be_completed(self):
        for updates in ({"errors": ["sampling_failed"]}, {"output_truncated": True}):
            with self.subTest(updates=updates):
                code, text, _ = self.fake_run(_completed_worker(), **updates)
                report = self.decoded(code, text, 27)
                self.assertTrue(all(report["cleanup"].values()))
                self.assertEqual(report["outcome"]["status"], "failed")

    def test_existing_output_is_never_overwritten_or_followed_by_capture(self):
        output = self.root / "existing.json"
        output.write_bytes(b"original evidence")
        with mock.patch.object(capture, "run_process") as run:
            code, stdout, stderr = self.invoke(self.arguments(output))
        self.assertEqual(code, 74)
        self.assertEqual(output.read_bytes(), b"original evidence")
        self.assertEqual(stdout, "")
        self.assertIn("binary_evidence_output_failed", stderr)
        run.assert_not_called()
        self.assertEqual(set(self.work.iterdir()), {self.neighbor})

    def test_new_output_file_contains_valid_completed_static_evidence(self):
        output = self.root / "new.json"
        with mock.patch.object(capture, "run_process", side_effect=self.fake_capture(_completed_worker())):
            code, stdout, stderr = self.invoke(self.arguments(output))
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        self.decoded(code, output.read_text(encoding="utf-8"), 0)

    def test_file_bytes_rejects_nonregular_input_before_open(self):
        with mock.patch.object(Path, "open", side_effect=AssertionError("nonregular input opened")) as opened:
            with self.assertRaisesRegex(binary.BinaryAuditError, "^input_not_regular$"):
                binary._file_bytes(self.inputs, 1024)
        opened.assert_not_called()

    def test_file_bytes_keeps_post_open_regularity_and_size_checks(self):
        expected = self.backend.read_bytes()
        self.assertEqual(binary._file_bytes(self.backend, len(expected)), expected)
        with self.assertRaisesRegex(binary.BinaryAuditError, "^input_size_limit$"):
            binary._file_bytes(self.backend, len(expected) - 1)
        with mock.patch.object(binary.os, "fstat", return_value=mock.Mock(st_mode=stat.S_IFDIR)):
            with self.assertRaisesRegex(binary.BinaryAuditError, "^input_not_regular$"):
                binary._file_bytes(self.backend, 1024)

    @unittest.skipUnless(os.name == "posix", "real leaf symlink fixture is POSIX-only")
    def test_file_bytes_rejects_existing_and_dangling_leaf_links_without_open(self):
        for name, target in (("link", self.backend), ("dangling", self.root / "missing")):
            link = self.root / name
            link.symlink_to(target)
            with self.subTest(name=name), mock.patch.object(
                    Path, "open", side_effect=AssertionError("leaf link opened")) as opened:
                with self.assertRaisesRegex(binary.BinaryAuditError, "^input_link_not_supported$"):
                    binary._file_bytes(link, 1024)
            opened.assert_not_called()

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "mkfifo"), "real FIFO fixture is POSIX-only")
    def test_fifo_worker_result_is_failed_without_blocking_or_retaining_scratch(self):
        capture_fixture = self.fake_capture(None)
        fifo_paths = set()
        original_open = Path.open

        def fake_with_fifo(command, **kwargs):
            result = capture_fixture(command, **kwargs)
            fifo = Path(command[4]).with_name("result.json")
            os.mkfifo(fifo)
            fifo_paths.add(fifo)
            return result

        def guarded_open(path, *args, **kwargs):
            # Fail immediately if the pre-open guard regresses: never let the
            # test itself block opening a FIFO with no writer.
            if path in fifo_paths:
                raise AssertionError("controller attempted to open FIFO result")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(capture, "run_process", side_effect=fake_with_fifo), \
                mock.patch.object(Path, "open", new=guarded_open):
            code, text, _ = self.invoke(self.arguments())
        report = self.decoded(code, text, 27)
        self.assertEqual(report["outcome"]["status"], "failed")
        self.assertTrue(all(report["cleanup"].values()))
        self.assertIn("invalid_binary_worker_result", report["diagnostics"])
        self.assertFalse(self.scratch_paths[0].exists())
        self.assertEqual(set(self.work.iterdir()), {self.neighbor})

    @unittest.skipUnless(sys.platform in ("win32", "linux"), "verified owned-tree containment requires Windows or Linux")
    def test_real_outer_controller_rejects_unpinned_inputs_without_cuda(self):
        # The supplied 'tool' is inert text. Its hash fails before any native
        # diagnostic invocation, while real Python children exercise containment.
        code, text, stderr = self.invoke(self.arguments(timeout=15))
        report = self.decoded(code, text, 23)
        self.assertEqual(stderr, "")
        self.assertEqual(report["outcome"]["status"], "rejected")
        self.assertIn("unpinned_cuobjdump", report["diagnostics"])
        self.assertTrue(all(report["cleanup"].values()))
        self.assertEqual(report["tools"]["invocations"], 0)
        self.assertEqual(set(self.work.iterdir()), {self.neighbor})


if __name__ == "__main__":
    unittest.main()
