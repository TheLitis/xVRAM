"""Read-only platform samples and owned-child containment without a CUDA driver."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from xvram.compat_audit_platform import (
    MAX_MODULES, PlatformObservationUnavailable, _parse_device_csv, _proc_module_paths,
    drain_windows_job, loaded_modules, sample_device, sample_process,
)


class AuditPlatformTests(unittest.TestCase):
    def test_module_snapshot_of_only_our_child(self):
        with subprocess.Popen([sys.executable, "-c", "import sys; print('ready',flush=True); sys.stdin.readline()"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as child:
            try:
                self.assertEqual(child.stdout.readline().strip(), b"ready")
                modules = loaded_modules(child)
                self.assertGreater(len(modules), 0)
                self.assertLessEqual(len(modules), MAX_MODULES)
                self.assertTrue(all(path.is_absolute() for path in modules))
                self.assertTrue(any("python" in path.name.lower() for path in modules))
                memory = sample_process(child)
                self.assertTrue(memory["available"], memory)
                self.assertEqual(memory["scope"], "direct_command_process")
                self.assertGreater(memory["rss_bytes"], 0)
                self.assertGreaterEqual(memory["peak_rss_bytes"], memory["rss_bytes"])
                self.assertNotIn("loaded_module_paths", memory)
            finally:
                child.communicate(b"\n", timeout=10)

    def test_module_failure_is_unknown_not_empty_verified(self):
        with subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL) as child:
            child.wait(timeout=10)
            with self.assertRaises(PlatformObservationUnavailable):
                loaded_modules(child)

    def test_proc_mapping_parser_bounds_and_escapes(self):
        records = (
            "1000-2000 r-xp 0000 08:01 1 /tmp/module\\040one.so\n"
            "2000-3000 r--p 0000 08:01 1 /tmp/module\\040one.so\n"
            "3000-4000 rw-p 0000 00:00 0 [heap]\n"
        )
        self.assertEqual(_proc_module_paths(records), [Path("/tmp/module one.so")])
        for invalid in ("", "malformed", "1000-2000 r-xp 0000 08:01 1 /tmp/gone.so (deleted)\n"):
            with self.assertRaises(PlatformObservationUnavailable):
                _proc_module_paths(invalid)
        with mock.patch("xvram.compat_audit_platform.MAX_MODULES", 1):
            with self.assertRaises(PlatformObservationUnavailable):
                _proc_module_paths(records + "4000-5000 r-xp 0000 08:01 2 /tmp/second.so\n")

    def test_device_csv_is_scoped_and_integer_bounded(self):
        value = _parse_device_csv('"NVIDIA Test, GPU", 8192, 1024, 7168, 616.56\n', 0)
        self.assertTrue(value["available"])
        self.assertEqual(value["scope"], "whole_device_not_process")
        self.assertEqual(value["total_bytes"], 8192 * 1024**2)
        for invalid in (
            "NVIDIA, 8192, N/A, 100, 616.56", "NVIDIA, 8192, 8193, 0, 616.56",
            "NVIDIA, -1, 0, 0, 616.56", "NVIDIA, 18446744073709551616, 0, 0, 616.56",
            "NVIDIA, 8192, 1, 1, 616.56\nNVIDIA, 8192, 1, 1, 616.56", "",
        ):
            with self.assertRaises(ValueError):
                _parse_device_csv(invalid, 0)

    def test_device_queries_do_not_require_a_driver_in_ci(self):
        with mock.patch("xvram.compat_audit_platform.shutil.which", return_value=None):
            result = sample_device()
        self.assertFalse(result["available"])
        self.assertIsNone(result["total_bytes"])
        self.assertEqual(result["scope"], "whole_device_not_process")
        with mock.patch("xvram.compat_audit_platform.subprocess.run", side_effect=subprocess.TimeoutExpired("sample", 2)):
            self.assertFalse(sample_device(executable="test-only-smi")["available"])
        completed = subprocess.CompletedProcess([], 0, stdout="NVIDIA Test, 8192, 1000, 7192, 616.56\n")
        with mock.patch("xvram.compat_audit_platform.subprocess.run", return_value=completed) as run:
            self.assertTrue(sample_device(executable="test-only-smi")["available"])
            self.assertIn("--id=0", run.call_args.args[0])
            self.assertNotIn("shell", run.call_args.kwargs)

    def test_invalid_timeouts_and_unknown_closed_job(self):
        for timeout in (float("nan"), float("inf"), -1, 61):
            with self.assertRaises(ValueError):
                sample_device(timeout_seconds=timeout)
            with self.assertRaises(ValueError):
                drain_windows_job(object(), timeout_seconds=timeout)
        self.assertFalse(drain_windows_job(object()))
        with self.assertRaises(ValueError):
            sample_device(device=True)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object containment")
    def test_job_drain_waits_for_our_child_and_grandchild(self):
        from xvram.torch_bench import _WindowsJob

        with tempfile.TemporaryDirectory(prefix="xvram-audit-job-") as raw:
            marker = Path(raw) / "grandchild.pid"
            code = (
                "import pathlib,subprocess,sys,time; sys.stdin.readline(); "
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                f"pathlib.Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(60)"
            )
            child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            job = None
            try:
                job = _WindowsJob(child)
                child.stdin.write(b"\n")
                child.stdin.flush()
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "owned grandchild did not start")
                grandchild_pid = int(marker.read_text())
                self.assertTrue(drain_windows_job(job, timeout_seconds=10))
                self.assertIsNotNone(job._handle, "caller must still own the open drained job")
                child.wait(timeout=10)
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
                kernel.OpenProcess.restype = ctypes.c_void_p
                kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                kernel.GetExitCodeProcess.restype = ctypes.c_int
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                kernel.CloseHandle.restype = ctypes.c_int
                handle = kernel.OpenProcess(0x1000, 0, grandchild_pid)
                if handle:
                    try:
                        status = ctypes.c_uint32()
                        self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(status)))
                        self.assertNotEqual(status.value, 259)
                    finally:
                        kernel.CloseHandle(handle)
            finally:
                if job is not None:
                    drain_windows_job(job, timeout_seconds=10)
                    job.close()
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
                child.stdin.close()


if __name__ == "__main__":
    unittest.main()
