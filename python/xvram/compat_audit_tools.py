"""Bounded native-tool I/O inside the existing audit controller's containment.

Production callers MUST execute this helper in a worker already owned by
``compat_audit_capture.run_process`` (Windows Job Object / POSIX process group).
This helper creates no Job, process group or session. It kills/reaps only its
direct tool process; the outer controller remains responsible for descendant
drain, including descendants retaining pipe handles after their parent exits.

Commands are argument vectors, never shell fragments. Raw returned bytes and
streamed lines are internal evidence and must be normalized before serialization.
Callbacks are trusted bounded CPU parsers: they must return promptly and must
not perform native/GPU submissions. The outer deadline covers a stuck parser.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable


_READ_BYTES = 16384
_OUTPUT_CAP = 256 * 1024 * 1024
_DRAIN_GRACE_SECONDS = 0.5
_REAP_GRACE_SECONDS = 2.0


class AuditToolError(RuntimeError):
    """Fixed diagnostic code; never contains argv, paths or native stderr."""
    def __init__(self, code, *, returncode=None, direct_child_reaped=False, pipes_drained=False):
        super().__init__(code)
        self.code = code
        self.returncode = returncode
        self.direct_child_reaped = direct_child_reaped
        self.pipes_drained = pipes_drained


@dataclass(frozen=True)
class ToolResult:
    stdout: bytes
    stderr: bytes
    returncode: int
    direct_child_reaped: bool
    pipes_drained: bool
    stdout_bytes: int
    stderr_bytes: int


def _read_available(pipe, amount):
    if os.name == "nt":
        import msvcrt
        # Borrow the existing pipe handle. With one reader, reading no more than
        # PeekNamedPipe's available bytes cannot wait on a descendant writer.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        peek = kernel.PeekNamedPipe
        peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                         ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        peek.restype = ctypes.c_int
        available = ctypes.c_uint32()
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(pipe.fileno()))
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            if ctypes.get_last_error() in (109, 232, 233):
                return b""  # Broken/disconnected writer, not a read failure.
            raise AuditToolError("tool_pipe_read_failed")
        if available.value == 0:
            return None
        return os.read(pipe.fileno(), min(amount, available.value))
    try:
        return os.read(pipe.fileno(), amount)
    except BlockingIOError:
        return None


def _validate(command, cwd, timeout_seconds, stdout_limit, stderr_limit, line_limit, consumer):
    if not isinstance(command, list) or not command or len(command) > 256 or any(
            not isinstance(value, str) or "\0" in value for value in command):
        raise AuditToolError("invalid_tool_command")
    if not command[0] or sum(len(value) for value in command) > 32768:
        raise AuditToolError("invalid_tool_command")
    if not isinstance(cwd, Path):
        raise AuditToolError("invalid_tool_directory")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 900 or not math.isfinite(timeout_seconds):
        raise AuditToolError("invalid_tool_timeout")
    for value in (stdout_limit, stderr_limit):
        if type(value) is not int or not 0 <= value <= _OUTPUT_CAP:
            raise AuditToolError("invalid_tool_output_limit")
    if type(line_limit) is not int or not 0 < line_limit <= 1024 * 1024:
        raise AuditToolError("invalid_tool_line_limit")
    if consumer is not None and not callable(consumer):
        raise AuditToolError("invalid_tool_consumer")


def run_tool(command: list[str], *, cwd: Path, timeout_seconds=30,
             stdout_limit=8 * 1024 * 1024, stderr_limit=64 * 1024,
             consume_stdout_line: Callable[[bytes], object] | None = None,
             line_limit=64 * 1024) -> ToolResult:
    """Run a direct diagnostic tool with bounded incremental stdout/stderr.

    Without a consumer, stdout is returned as bytes. With a consumer, each line
    (including its newline, if present) is delivered once and stdout is ``b""``.
    ``stdout_limit`` bounds TOTAL bytes even when lines are discarded; use e.g.
    128 MiB for a streamed ELF dump. ``line_limit`` applies to streamed lines,
    including a newline. A final unterminated line is delivered after EOF.

    Success requires exit 0, both pipe EOFs and a reaped direct process. Errors
    expose only a fixed code and cleanup status. No descendant cleanup is claimed.
    """
    _validate(command, cwd, timeout_seconds, stdout_limit, stderr_limit, line_limit, consume_stdout_line)
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   bufsize=0, close_fds=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        raise AuditToolError("tool_start_failed") from None
    output = {"stdout": bytearray(), "stderr": bytearray()}
    totals = {"stdout": 0, "stderr": 0}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    pipes = {"stdout": process.stdout, "stderr": process.stderr}
    open_pipes = set(pipes)
    pending_line = bytearray()
    failure = None
    reaped = drained = False
    exited_at = None

    def consume(line):
        try:
            consume_stdout_line(line)
        except Exception:
            raise AuditToolError("tool_stdout_consumer_failed") from None

    try:
        if os.name != "nt":
            for pipe in pipes.values():
                os.set_blocking(pipe.fileno(), False)
        while True:
            now = time.monotonic()
            if now - started >= timeout_seconds:
                raise AuditToolError("tool_timeout")
            if process.poll() is not None and exited_at is None:
                exited_at = now
            if exited_at is not None and open_pipes and now - exited_at >= _DRAIN_GRACE_SECONDS:
                raise AuditToolError("tool_pipe_drain_timeout")
            progressed = False
            for name in ("stdout", "stderr"):
                if name not in open_pipes:
                    continue
                chunk = _read_available(pipes[name], _READ_BYTES)
                if chunk is None:
                    continue
                if not chunk:
                    open_pipes.remove(name)
                    if name == "stdout" and consume_stdout_line is not None and pending_line:
                        consume(bytes(pending_line))
                        pending_line.clear()
                    continue
                progressed = True
                totals[name] += len(chunk)
                if totals[name] > limits[name]:
                    raise AuditToolError("tool_" + name + "_limit")
                if name != "stdout" or consume_stdout_line is None:
                    output[name].extend(chunk)
                    continue
                pending_line.extend(chunk)
                offset = 0
                while (end := pending_line.find(b"\n", offset)) != -1:
                    end += 1
                    if end - offset > line_limit:
                        raise AuditToolError("tool_stdout_line_limit")
                    consume(bytes(pending_line[offset:end]))
                    offset = end
                if offset:
                    del pending_line[:offset]
                if len(pending_line) > line_limit:
                    raise AuditToolError("tool_stdout_line_limit")
            if not open_pipes and process.poll() is not None:
                if time.monotonic() - started >= timeout_seconds:
                    raise AuditToolError("tool_timeout")
                drained = True
                break
            if not progressed:
                time.sleep(min(0.01, max(0, timeout_seconds - (time.monotonic() - started))))
        if process.returncode != 0:
            raise AuditToolError("tool_nonzero_exit")
    except AuditToolError as error:
        failure = error
    except Exception:
        failure = AuditToolError("tool_io_failed")
    finally:
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass  # A racing natural exit must still reach the independent wait.
        try:
            process.wait(timeout=_REAP_GRACE_SECONDS)
            reaped = True
        except (OSError, subprocess.SubprocessError):
            if failure is None:
                failure = AuditToolError("tool_reap_failed")
        for pipe in pipes.values():
            try:
                pipe.close()
            except OSError:
                if failure is None:
                    failure = AuditToolError("tool_pipe_close_failed")
        if os.name == "nt" and reaped:
            try:
                # Popen.wait does not close its owned Windows process handle.
                # Error tracebacks can retain Popen until cyclic GC; release it
                # explicitly after reap so a fresh extraction cwd is reusable.
                process._handle.Close()
            except OSError:
                if failure is None:
                    failure = AuditToolError("tool_process_handle_close_failed")
    if failure is not None:
        raise AuditToolError(failure.code, returncode=process.returncode,
                            direct_child_reaped=reaped, pipes_drained=drained) from None
    return ToolResult(bytes(output["stdout"]), bytes(output["stderr"]), process.returncode,
                      reaped, drained, totals["stdout"], totals["stderr"])
