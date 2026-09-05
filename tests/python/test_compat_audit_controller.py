from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

from xvram.compat_audit_capture import _terminate, parse_timings, run_process, sanitize_log
from xvram.compat_audit import _parser, build_command, clean_capture_environment, verify_model


class AuditControllerTests(unittest.TestCase):
    def test_success_output_is_redacted_and_reaped(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            result = run_process([sys.executable, "-c", "print('value 0xabcdef1234')"], output_dir=directory, timeout_seconds=15)
            self.assertEqual(result["exit_code"], 0)
            self.assertTrue(result["controller_reaped"])
            self.assertTrue(result["process_tree_drained"])
            self.assertNotIn("0xabcdef", (directory / "stdout.txt").read_text())
            self.assertIsNone(result["ttft_ms"])

    def test_crash_is_not_success(self):
        with tempfile.TemporaryDirectory() as raw:
            result = run_process([sys.executable, "-c", "raise SystemExit(17)"], output_dir=Path(raw), timeout_seconds=15)
            self.assertEqual(result["exit_code"], 17)
            self.assertTrue(result["controller_reaped"])

    def test_deadline_reaps_child_tree(self):
        with tempfile.TemporaryDirectory() as raw:
            pidfile = Path(raw) / "child.pid"
            code = "import os,time,pathlib; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(pidfile)
            result = run_process([sys.executable, "-c", code], output_dir=Path(raw), timeout_seconds=2)
            self.assertEqual(result["exit_code"], 26)
            self.assertTrue(result["timed_out"])
            self.assertTrue(result["controller_reaped"])
            if pidfile.exists():
                pid = int(pidfile.read_text())
                if os.name == "nt":
                    api = ctypes.WinDLL("kernel32", use_last_error=True)
                    api.OpenProcess.restype = ctypes.c_void_p
                    handle = api.OpenProcess(0x1000, False, pid)
                    if handle:
                        status = ctypes.c_ulong()
                        api.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(status))
                        api.CloseHandle(ctypes.c_void_p(handle))
                        self.assertNotEqual(status.value, 259)
                else:
                    # Orphaned children may briefly be zombies until init reaps;
                    # a live/sleeping child is never acceptable.
                    status = Path(f"/proc/{pid}/stat")
                    if status.exists():
                        self.assertEqual(status.read_text().split()[2], "Z")

    def test_missing_executable_reports_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            result = run_process([str(Path(raw) / "missing")], output_dir=Path(raw), timeout_seconds=15)
            self.assertEqual(result["exit_code"], 27)
            self.assertTrue(result["controller_reaped"])

    def test_bounded_output(self):
        with tempfile.TemporaryDirectory() as raw:
            result = run_process([sys.executable, "-c", "import sys;sys.stdout.write('x'*(9*1024*1024))"],
                                 output_dir=Path(raw), timeout_seconds=15)
            self.assertTrue(result["output_truncated"])
            self.assertLessEqual((Path(raw) / "stdout.txt").stat().st_size, 8 * 1024 * 1024)

    def test_timings_are_observations_not_invented(self):
        parsed = parse_timings("load time = 50.2 ms\nprompt eval time = 25.0 ms / 10 tokens (2.5 ms per token, 400.0 tokens per second)\neval time = 100.0 ms / 5 runs (20.0 ms per token, 50.0 tokens per second)")
        self.assertEqual(parsed["load_ms"], 50.2)
        self.assertEqual(parsed["prefill_tokens"], 10)
        self.assertEqual(parsed["decode_tokens_per_second"], 50.0)
        self.assertTrue(all(value is None for value in parse_timings("").values()))

    def test_profile_flags_do_not_allow_silent_offload_or_graphs(self):
        args = _parser().parse_args(["--binary-dir", ".", "--microbatch", "1"])
        command = build_command(args, Path("test.gguf"))
        for option, value in (("--fit", "off"), ("--flash-attn", "off"),
                              ("--cache-type-k", "f16"), ("--ubatch-size", "1"), ("--gpu-layers", "8")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("--no-context-shift", command)
        self.assertIn("--no-conversation", command)

    def test_environment_removes_external_profile_overrides(self):
        with mock.patch.dict(os.environ, {"GGML_BACKEND_PATH": "unapproved.dll", "LLAMA_ARG_FIT": "on",
                                          "CUDA_INJECTION64_PATH": "unexpected.dll", "NORMAL_AUDIT_TEST": "ok"}):
            env = clean_capture_environment()
            self.assertNotIn("GGML_BACKEND_PATH", env)
            self.assertNotIn("CUDA_INJECTION64_PATH", env)
            self.assertNotIn("LLAMA_ARG_FIT", env)
            self.assertEqual(env["NORMAL_AUDIT_TEST"], "ok")

    def test_wrong_model_hash_never_overwrites(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "weight.gguf"
            path.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                verify_model(Path(raw), {"files": [{"name": path.name, "bytes": 8, "sha256": "0" * 64}]})
            self.assertEqual(path.read_bytes(), b"existing")

    def test_log_privacy(self):
        self.assertEqual(sanitize_log("\x1b[31mstream=0xdeadbeef\x1b[0m"), "stream=[native-value-redacted]")
        self.assertEqual(sanitize_log("mapping 00007FFE1234ABCD"), "mapping [native-value-redacted]")
        self.assertEqual(sanitize_log("stream=0XABCDEF1234"), "stream=[native-value-redacted]")

    def test_nonfinite_timeout_never_starts_bootstrap(self):
        with mock.patch("xvram.compat_audit_capture.subprocess.Popen") as start:
            for value in (float("nan"), float("inf"), 0, -1, 86401):
                with self.assertRaises(ValueError):
                    run_process([sys.executable, "-c", "pass"], output_dir=Path("."), timeout_seconds=value)
            start.assert_not_called()

    def test_failed_job_drain_still_closes_job_and_reaps(self):
        job = mock.Mock()
        worker = mock.Mock()
        with mock.patch("xvram.compat_audit_capture.drain_windows_job", return_value=False):
            self.assertFalse(_terminate(worker, job))
        job.close.assert_called_once()
        worker.wait.assert_called_once_with(timeout=10)

    def test_job_drain_exception_and_wait_timeout_preserve_cleanup(self):
        job = mock.Mock()
        worker = mock.Mock()
        worker.wait.side_effect = [subprocess.TimeoutExpired("owned-worker", 10), 0]
        with mock.patch("xvram.compat_audit_capture.drain_windows_job", side_effect=OSError("query failed")):
            self.assertFalse(_terminate(worker, job))
        job.close.assert_called_once()
        worker.kill.assert_called_once()
        self.assertEqual(worker.wait.call_args_list, [mock.call(timeout=10), mock.call(timeout=2)])

    @unittest.skipUnless(os.name == "nt", "Windows bootstrap assignment failure")
    def test_job_assignment_failure_never_starts_native_command(self):
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "must-not-start.txt"
            code = f"import pathlib;pathlib.Path({str(marker)!r}).write_text('started')"
            with mock.patch("xvram.torch_bench._WindowsJob", side_effect=OSError("assignment failed")):
                result = run_process([sys.executable, "-c", code], output_dir=Path(raw), timeout_seconds=10)
            self.assertEqual(result["exit_code"], 27)
            self.assertTrue(result["controller_reaped"])
            self.assertTrue(result["process_tree_drained"])
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name != "nt", "POSIX leader PID ownership")
    def test_never_signals_a_previously_reaped_group_leader(self):
        worker = mock.Mock(returncode=0, pid=123456)
        with mock.patch("xvram.compat_audit_capture.os.killpg") as signal_group:
            self.assertFalse(_terminate(worker, None))
        signal_group.assert_not_called()

    @unittest.skipUnless(os.name != "nt", "POSIX unknown group observation")
    def test_unknown_group_drain_is_not_reported_as_success(self):
        worker = mock.Mock(returncode=None, pid=123456)
        with mock.patch("xvram.compat_audit_capture.os.killpg") as signal_group, \
             mock.patch("xvram.compat_audit_capture._posix_group_has_live_members", return_value=None):
            self.assertFalse(_terminate(worker, None))
        signal_group.assert_called_once()
        worker.wait.assert_called_once_with(timeout=10)

    def test_normal_exit_also_drains_background_descendant(self):
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "descendant.pid"
            code = (
                "import pathlib,subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))"
            )
            result = run_process([sys.executable, "-c", code], output_dir=Path(raw), timeout_seconds=15)
            self.assertEqual(result["exit_code"], 0, result)
            self.assertTrue(result["process_tree_drained"], result)
            self.assertTrue(result["controller_reaped"])
            self.assertTrue(marker.exists())
            pid = int(marker.read_text())
            if os.name == "nt":
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
                kernel.OpenProcess.restype = ctypes.c_void_p
                handle = kernel.OpenProcess(0x1000, False, pid)
                if handle:
                    status = ctypes.c_uint32()
                    try:
                        self.assertTrue(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(status)))
                        self.assertNotEqual(status.value, 259)
                    finally:
                        kernel.CloseHandle(ctypes.c_void_p(handle))
            else:
                status = Path(f"/proc/{pid}/stat")
                if status.exists():
                    self.assertIn(status.read_text().split(")", 1)[1].split()[0], ("Z", "X"))


if __name__ == "__main__":
    unittest.main()
