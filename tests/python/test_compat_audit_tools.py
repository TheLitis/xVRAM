"""CPU-only bounded tool I/O, direct-child reap and outer containment tests."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from xvram.compat_audit_capture import run_process
from xvram.compat_audit_tools import AuditToolError, run_tool


class AuditToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def command(self, code):
        return [sys.executable, "-c", code]

    def fake_process(self):
        process = mock.Mock(returncode=0)
        process.poll.return_value = 0
        process.wait.return_value = 0
        process.stdout.fileno.return_value = 101
        process.stderr.fileno.return_value = 102
        return process

    def run_fake(self, process, *, read_error=None):
        with mock.patch("xvram.compat_audit_tools.subprocess.Popen", return_value=process), \
                mock.patch("xvram.compat_audit_tools._read_available", return_value=b"", side_effect=read_error), \
                mock.patch("xvram.compat_audit_tools.os.set_blocking"):
            return run_tool(self.command("pass"), cwd=self.directory)

    def test_small_output_and_direct_process_are_drained_and_reaped(self):
        started = []
        real_popen = subprocess.Popen

        def start(*args, **kwargs):
            self.assertFalse(kwargs.get("start_new_session", False))
            self.assertNotIn("process_group", kwargs)
            self.assertFalse(kwargs.get("shell", False))
            child = real_popen(*args, **kwargs)
            started.append(child)
            return child

        with mock.patch("xvram.compat_audit_tools.subprocess.Popen", side_effect=start):
            result = run_tool(self.command("import os;os.write(1,b'one\\ntwo');os.write(2,b'warning')"), cwd=self.directory)
        self.assertEqual(result.stdout, b"one\ntwo")
        self.assertEqual(result.stderr, b"warning")
        self.assertEqual((result.stdout_bytes, result.stderr_bytes), (7, 7))
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.direct_child_reaped)
        self.assertTrue(result.pipes_drained)
        self.assertEqual(started[0].wait(timeout=0), 0)
        self.assertTrue(started[0].stdout.closed and started[0].stderr.closed)

    def test_streamed_lines_are_incremental_with_final_unterminated_line(self):
        lines = []
        code = "import os;os.write(1,b'alpha\\nbeta\\r\\ngamma');os.write(2,b'notice')"
        result = run_tool(self.command(code), cwd=self.directory, consume_stdout_line=lines.append)
        self.assertEqual(lines, [b"alpha\n", b"beta\r\n", b"gamma"])
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stdout_bytes, len(b"alpha\nbeta\r\ngamma"))
        self.assertEqual(result.stderr, b"notice")

    def test_streaming_does_not_retain_discarded_dump(self):
        count = 0

        def consume(line):
            nonlocal count
            self.assertEqual(len(line), 65)
            count += 1

        code = "import os;data=b'x'*64+b'\\n';[os.write(1,data*256) for _ in range(32)]"
        result = run_tool(self.command(code), cwd=self.directory, consume_stdout_line=consume, stdout_limit=1024 * 1024)
        self.assertEqual(count, 8192)
        self.assertEqual(result.stdout_bytes, 8192 * 65)
        self.assertEqual(result.stdout, b"")

    def test_total_stdout_bound_applies_even_to_discarded_lines(self):
        with self.assertRaises(AuditToolError) as raised:
            run_tool(self.command("import os;os.write(1,b'x\\n'*10000)"), cwd=self.directory,
                     stdout_limit=100, consume_stdout_line=lambda _: None)
        self.assertEqual(raised.exception.code, "tool_stdout_limit")
        self.assertTrue(raised.exception.direct_child_reaped)

    def test_independent_stderr_and_stdout_limits(self):
        for endpoint, name in ((1, "stdout"), (2, "stderr")):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(AuditToolError) as raised:
                    run_tool(self.command(f"import os;os.write({endpoint},b'x'*1000)"), cwd=self.directory,
                             stdout_limit=128, stderr_limit=128)
                self.assertEqual(raised.exception.code, "tool_" + name + "_limit")
                self.assertTrue(raised.exception.direct_child_reaped)

    def test_long_and_unterminated_lines_are_bounded(self):
        for output in (b"x" * 33, b"x" * 32 + b"\n"):
            with self.subTest(output=output):
                with self.assertRaises(AuditToolError) as raised:
                    run_tool(self.command(f"import os;os.write(1,{output!r})"), cwd=self.directory,
                             consume_stdout_line=lambda _: None, line_limit=32)
                self.assertEqual(raised.exception.code, "tool_stdout_line_limit")
                self.assertTrue(raised.exception.direct_child_reaped)

    def test_timeout_kills_and_reaps_only_direct_child(self):
        started = time.monotonic()
        with self.assertRaises(AuditToolError) as raised:
            run_tool(self.command("import time;time.sleep(60)"), cwd=self.directory, timeout_seconds=0.1)
        self.assertEqual(raised.exception.code, "tool_timeout")
        self.assertTrue(raised.exception.direct_child_reaped)
        self.assertLess(time.monotonic() - started, 5)

    def test_nonzero_and_consumer_errors_have_no_native_text(self):
        code = "import sys;sys.stderr.write('C:/private/0xabcdef1234');sys.exit(17)"
        with self.assertRaises(AuditToolError) as raised:
            run_tool(self.command(code), cwd=self.directory)
        self.assertEqual(str(raised.exception), "tool_nonzero_exit")
        self.assertEqual(raised.exception.returncode, 17)
        self.assertTrue(raised.exception.direct_child_reaped)
        self.assertTrue(raised.exception.pipes_drained)

        def consume(_):
            raise ValueError("C:/private/0xabcdef1234")

        with self.assertRaises(AuditToolError) as raised:
            run_tool(self.command("print('line')"), cwd=self.directory, consume_stdout_line=consume)
        self.assertEqual(str(raised.exception), "tool_stdout_consumer_failed")
        self.assertTrue(raised.exception.direct_child_reaped)

    def test_missing_tool_is_sanitized(self):
        with self.assertRaises(AuditToolError) as raised:
            run_tool([str(self.directory / "private-missing-tool")], cwd=self.directory)
        self.assertEqual(str(raised.exception), "tool_start_failed")
        self.assertNotIn(str(self.directory), repr(raised.exception))

    def test_invalid_configuration_does_not_start_process(self):
        configurations = ({"timeout_seconds": float("nan")}, {"timeout_seconds": float("inf")},
                          {"timeout_seconds": False}, {"timeout_seconds": 0}, {"timeout_seconds": 10 ** 1000},
                          {"stdout_limit": -1}, {"stderr_limit": 257 * 1024 * 1024},
                          {"stdout_limit": True}, {"line_limit": 0}, {"consume_stdout_line": 4})
        with mock.patch("xvram.compat_audit_tools.subprocess.Popen") as start:
            for configuration in configurations:
                with self.subTest(configuration=configuration), self.assertRaises(AuditToolError):
                    run_tool(self.command("pass"), cwd=self.directory, **configuration)
            for command in ([], [""], "shell fragment", [sys.executable, "bad\0arg"]):
                with self.subTest(command=command), self.assertRaises(AuditToolError):
                    run_tool(command, cwd=self.directory)
            start.assert_not_called()

    def test_injected_pipe_read_failure_still_reaps_and_closes_every_pipe(self):
        for error, expected in ((OSError("C:/private/0xabcdef"), "tool_io_failed"),
                                (AuditToolError("tool_pipe_read_failed"), "tool_pipe_read_failed")):
            with self.subTest(error=type(error).__name__):
                process = self.fake_process()
                with self.assertRaises(AuditToolError) as raised:
                    self.run_fake(process, read_error=error)
                self.assertEqual(str(raised.exception), expected)
                self.assertTrue(raised.exception.direct_child_reaped)
                self.assertFalse(raised.exception.pipes_drained)
                process.wait.assert_called_once_with(timeout=2.0)
                process.stdout.close.assert_called_once_with()
                process.stderr.close.assert_called_once_with()
                if os.name == "nt":
                    process._handle.Close.assert_called_once_with()

    def test_injected_reap_failure_is_not_reported_as_success(self):
        process = self.fake_process()
        process.wait.side_effect = subprocess.TimeoutExpired("C:/private/0xabcdef", 2)
        with self.assertRaises(AuditToolError) as raised:
            self.run_fake(process)
        self.assertEqual(str(raised.exception), "tool_reap_failed")
        self.assertFalse(raised.exception.direct_child_reaped)
        self.assertTrue(raised.exception.pipes_drained)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()
        process._handle.Close.assert_not_called()

    def test_injected_close_failure_continues_independent_cleanup_stages(self):
        process = self.fake_process()
        process.stdout.close.side_effect = OSError("C:/private/0xabcdef")
        with self.assertRaises(AuditToolError) as raised:
            self.run_fake(process)
        self.assertEqual(str(raised.exception), "tool_pipe_close_failed")
        self.assertTrue(raised.exception.direct_child_reaped)
        self.assertTrue(raised.exception.pipes_drained)
        process.stderr.close.assert_called_once_with()
        if os.name == "nt":
            process._handle.Close.assert_called_once_with()

        process = self.fake_process()
        process.wait.side_effect = subprocess.TimeoutExpired("private", 2)
        process.stdout.close.side_effect = OSError("private")
        process.stderr.close.side_effect = OSError("private")
        with self.assertRaises(AuditToolError) as raised:
            self.run_fake(process, read_error=OSError("private"))
        self.assertEqual(str(raised.exception), "tool_io_failed")
        self.assertFalse(raised.exception.direct_child_reaped)
        self.assertFalse(raised.exception.pipes_drained)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    def test_injected_kill_race_does_not_skip_independent_wait(self):
        process = self.fake_process()
        process.poll.return_value = None
        process.kill.side_effect = OSError("racing natural exit")
        with self.assertRaises(AuditToolError) as raised:
            self.run_fake(process, read_error=AuditToolError("tool_timeout"))
        self.assertEqual(str(raised.exception), "tool_timeout")
        self.assertTrue(raised.exception.direct_child_reaped)
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=2.0)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    @unittest.skipUnless(os.name == "nt", "Windows owned process handle")
    def test_injected_process_handle_close_failure_is_not_silent(self):
        process = self.fake_process()
        process._handle.Close.side_effect = OSError("C:/private/0xabcdef")
        with self.assertRaises(AuditToolError) as raised:
            self.run_fake(process)
        self.assertEqual(str(raised.exception), "tool_process_handle_close_failed")
        self.assertTrue(raised.exception.direct_child_reaped)
        self.assertTrue(raised.exception.pipes_drained)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    def test_inherited_descendant_pipes_are_bounded_and_outer_controller_drains(self):
        # Only this fixture creates descendants. Reuse the project's real outer
        # containment; the tool runner must never create a competing Job/group.
        native = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])"
        worker = (
            "from pathlib import Path\n"
            "import sys\n"
            "from xvram.compat_audit_tools import run_tool,AuditToolError\n"
            "try:\n"
            f"    run_tool([sys.executable,'-c',{native!r}],cwd=Path('.'),timeout_seconds=5)\n"
            "except AuditToolError as error:\n"
            "    assert error.code=='tool_pipe_drain_timeout',error.code\n"
            "    assert error.direct_child_reaped\n"
            "    assert not error.pipes_drained\n"
            "    print('bounded tool pipe drain')\n"
            "else:\n"
            "    raise AssertionError('inherited writer reported success')\n"
        )
        # The existing outer controller's process-object disposal is not this
        # helper's lifecycle contract; do not make its transient Windows cwd lock
        # part of cleanup of this test's output directory.
        result = run_process(self.command(worker), output_dir=self.directory, timeout_seconds=15)
        self.assertEqual(result["exit_code"], 0, result)
        self.assertTrue(result["controller_reaped"])
        self.assertTrue(result["process_tree_drained"])
        self.assertIn("bounded tool pipe drain", (self.directory / "stdout.txt").read_text())


if __name__ == "__main__":
    unittest.main()
