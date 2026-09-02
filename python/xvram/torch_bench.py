"""Controller/worker benchmark for lease-scoped PyTorch inference.

The public process owns all files and presentation.  CUDA and PyTorch work is
isolated in a child process so a driver failure, protocol violation, or hang can
be converted into a schema-valid report without endangering the controller.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Optional, Sequence, Union

from .torch_protocol import Frame, MessageType, ProtocolError, read_frame, write_frame
from .torch_report import (
    EXIT_COMPLETED,
    EXIT_CORRUPTION,
    EXIT_INTERNAL,
    EXIT_OUTPUT,
    EXIT_PREREQUISITE,
    EXIT_PRESSURE,
    EXIT_RUNTIME,
    EXIT_TIMEOUT,
    EXIT_USAGE,
    dumps,
    empty_report,
    finalize_proof,
    success_semantics,
    validate_report_envelope,
    validate_trace_record,
)


DEFAULT_SEED = 0x585652414D503034
_IPC_QUEUE_FRAMES = 256
_MAX_TRACE_RECORDS = 100_000
_MAX_STDERR_BYTES = 1024 * 1024
_SIZE_SUFFIXES = {
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
}


class BenchError(RuntimeError):
    def __init__(self, exit_code: int, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stage = stage
        self.code = code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def parse_size(value: str, *, allow_auto: bool = False) -> Union[int, str]:
    raw = value.strip().lower()
    if allow_auto and raw == "auto":
        return "auto"
    split = 0
    while split < len(raw) and (raw[split].isdigit() or raw[split] == "."):
        split += 1
    if split == 0:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    number_text, suffix = raw[:split], raw[split:] or "b"
    if suffix not in _SIZE_SUFFIXES:
        raise argparse.ArgumentTypeError(f"unknown size suffix in {value}")
    try:
        number = float(number_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {value}") from exc
    result = number * _SIZE_SUFFIXES[suffix]
    if not math.isfinite(result) or number <= 0 or result < 1 or result > (2**63 - 1):
        raise argparse.ArgumentTypeError(f"size out of range: {value}")
    parsed = int(result)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"size out of range: {value}")
    return parsed


def _normalize_plan(args: argparse.Namespace) -> dict[str, Any]:
    headroom = int(parse_size(args.device_headroom))
    cache_raw = parse_size(args.cache_target, allow_auto=True)
    # Zero is the native runtime's sentinel for a live-budget-derived target. The controller
    # never initializes CUDA; only the isolated worker may inspect CUDA/WDDM state.
    cache_target = 0 if cache_raw == "auto" else int(cache_raw)
    return {
        "device": args.device,
        "model": args.model,
        "model_ratio": args.model_ratio,
        "layers": args.layers,
        "batch": args.batch,
        "sequence": args.sequence,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "heads": args.heads,
        "dtype": args.dtype,
        "policy": args.policy,
        "cache_target_bytes": cache_target,
        "chunk_size_bytes": int(parse_size(args.chunk_size)),
        "device_headroom_bytes": headroom,
        "scratch_cap_bytes": int(parse_size(args.scratch_cap)),
        "prefetch_distance": args.prefetch_distance,
        "sdpa_backend": args.sdpa_backend,
        "seed": f"0x{args.seed:016x}",
        "include_identifiers": bool(args.include_identifiers),
    }


class _WindowsJob:
    """Kill-on-close job object used to contain the worker on Windows."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        pass

    _EXTENDED_LIMIT_INFORMATION._fields_ = [
        ("BasicLimitInformation", _BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        self._kernel32 = kernel32
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = self._EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            ctypes.c_void_p(self._handle),
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            self.close()
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(self._handle), ctypes.c_void_p(int(process._handle))
        ):
            self.close()
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._kernel32.CloseHandle(ctypes.c_void_p(handle))
            self._handle = None


@dataclass
class ControllerResult:
    report: dict[str, Any]
    exit_code: int
    trace_records: list[dict[str, Any]]
    stderr: str


def _reader(stream: BinaryIO, messages: queue.Queue[Union[Frame, BaseException]]) -> None:
    try:
        while True:
            messages.put(read_frame(stream))
    except BaseException as exc:
        messages.put(exc)


def _stderr_reader(stream: BinaryIO, tail: bytearray) -> None:
    """Drain worker stderr without allowing diagnostics to exhaust controller RAM."""

    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            tail.extend(chunk)
            overflow = len(tail) - _MAX_STDERR_BYTES
            if overflow > 0:
                del tail[:overflow]
    except (OSError, ValueError):
        return


def _terminate_worker(process: subprocess.Popen[bytes], job: Optional[_WindowsJob]) -> None:
    if process.poll() is None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)
    if process.poll() is not None:
        if job is not None:
            job.close()
        return
    if os.name == "nt":
        if job is not None:
            job.close()
        else:
            process.kill()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def _protocol_sequence(payload: Mapping[str, Any], previous: int) -> int:
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= previous:
        raise ValueError("worker protocol sequence is missing or not strictly increasing")
    return sequence


def _retired_progress(payload: Mapping[str, Any], previous: int) -> int:
    retired = payload.get("operations_retired")
    if not isinstance(retired, int) or isinstance(retired, bool) or retired <= previous:
        raise ValueError(
            "progress must contain a strictly increasing positive operations_retired count"
        )
    return retired


def run_controller(
    plan: Mapping[str, Any],
    *,
    timeout_seconds: int = 900,
    stall_timeout_seconds: int = 15,
    worker_command: Optional[Sequence[str]] = None,
) -> ControllerResult:
    command = list(worker_command or (sys.executable, "-m", "xvram.torch_bench", "--worker"))
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process: subprocess.Popen[bytes] = subprocess.Popen(command, **popen_kwargs)
    except OSError as exc:
        report = empty_report(
            plan,
            exit_code=EXIT_RUNTIME,
            status="failed",
            message=str(exc),
            include_identifiers=bool(plan.get("include_identifiers", False)),
        )
        report["diagnostics"].append(
            {"stage": "controller", "code": "worker_spawn_failure", "message": str(exc)}
        )
        report["cleanup"]["worker_reaped"] = True
        marker = {
            "schema_version": 1,
            "record_type": "xvram.pytorch_trace",
            "sequence": 1,
            "monotonic_ns": time.monotonic_ns(),
            "kind": "verification",
            "region_id": None,
            "allocation_id": None,
            "operation": "trace_incomplete",
            "bytes": 0,
            "reason": "worker was not created",
        }
        report["execution"]["trace_records"] = 1
        finalize_proof(report)
        return ControllerResult(report, EXIT_RUNTIME, [marker], "")
    job: Optional[_WindowsJob] = None
    stderr_tail = bytearray()
    stderr_thread: Optional[threading.Thread] = None
    if process.stderr is not None:
        stderr_thread = threading.Thread(
            target=_stderr_reader,
            args=(process.stderr, stderr_tail),
            daemon=True,
        )
        stderr_thread.start()
    final_report: Optional[dict[str, Any]] = None
    candidate_report: Optional[dict[str, Any]] = None
    received_final = False
    trace_records: list[dict[str, Any]] = []
    try:
        if os.name == "nt":
            job = _WindowsJob(process)
        assert process.stdin is not None
        assert process.stdout is not None
        write_frame(process.stdin, MessageType.PLAN, plan)
        process.stdin.close()

        # A bounded queue preserves pipe backpressure when a faulty worker floods
        # heartbeat/trace frames faster than the controller can validate them.
        messages: queue.Queue[Union[Frame, BaseException]] = queue.Queue(
            maxsize=_IPC_QUEUE_FRAMES
        )
        threading.Thread(target=_reader, args=(process.stdout, messages), daemon=True).start()
        started = time.monotonic()
        last_retired_progress = started
        last_protocol_sequence = 0
        last_trace_sequence = 0
        operations_retired = 0
        failure: Optional[str] = None
        timed_out = False

        while candidate_report is None and failure is None:
            now = time.monotonic()
            if (
                now - started >= timeout_seconds
                or now - last_retired_progress >= stall_timeout_seconds
            ):
                timed_out = True
                failure = (
                    "worker exceeded the overall deadline"
                    if now - started >= timeout_seconds
                    else "worker stopped retiring operations"
                )
                break
            try:
                wait_budget = min(
                    1.0,
                    timeout_seconds - (now - started),
                    stall_timeout_seconds - (now - last_retired_progress),
                )
                item = messages.get(timeout=max(0.001, wait_budget))
            except queue.Empty:
                if process.poll() is not None:
                    failure = f"worker exited before final report (exit {process.returncode})"
                    break
                continue
            if isinstance(item, BaseException):
                if process.poll() is not None:
                    failure = f"worker protocol ended before final report: {item}"
                    break
                failure = f"worker protocol failure: {item}"
                break
            try:
                last_protocol_sequence = _protocol_sequence(
                    item.payload, last_protocol_sequence
                )
            except ValueError as exc:
                failure = str(exc)
                break
            if item.message_type is MessageType.TRACE:
                records = item.payload.get("records", [])
                if not isinstance(records, list) or not all(isinstance(x, dict) for x in records):
                    failure = "worker sent an invalid trace batch"
                    break
                try:
                    for record in records:
                        validate_trace_record(record)
                        sequence = int(record["sequence"])
                        if sequence <= last_trace_sequence:
                            raise ValueError(
                                "trace sequence is not strictly increasing"
                            )
                        last_trace_sequence = sequence
                except ValueError as exc:
                    failure = f"worker sent an invalid trace record: {exc}"
                    break
                if len(trace_records) + len(records) > _MAX_TRACE_RECORDS:
                    failure = "worker trace exceeded the controller record limit"
                    break
                trace_records.extend(records)
            elif item.message_type is MessageType.FINAL:
                candidate = item.payload.get("report")
                if not isinstance(candidate, dict):
                    failure = "worker final frame did not contain a report object"
                    break
                try:
                    validate_report_envelope(candidate)
                except ValueError as exc:
                    failure = f"worker final report failed validation: {exc}"
                    break
                if int(candidate["execution"]["trace_records"]) != len(trace_records):
                    failure = "worker final report did not reconcile its trace record count"
                    break
                if (
                    int(candidate["outcome"]["exit_code"]) == EXIT_COMPLETED
                    and not trace_records
                ):
                    failure = "worker completed without a trace record"
                    break
                if (
                    int(candidate["outcome"]["exit_code"]) == EXIT_COMPLETED
                    and operations_retired < int(candidate["execution"]["regions_completed"])
                ):
                    failure = "worker progress did not reconcile completed graph regions"
                    break
                candidate_report = candidate
            elif item.message_type is MessageType.PROGRESS:
                try:
                    operations_retired = _retired_progress(
                        item.payload, operations_retired
                    )
                except ValueError as exc:
                    failure = f"worker sent invalid progress: {exc}"
                    break
                last_retired_progress = time.monotonic()
            elif item.message_type is not MessageType.HEARTBEAT:
                failure = f"unexpected worker message {item.message_type.name}"
                break

        if candidate_report is not None and failure is None:
            now = time.monotonic()
            # Once FINAL has arrived, no further progress frame is expected. Give
            # the worker a short, independent grace period to flush and exit, still
            # bounded by the overall controller deadline.
            exit_wait = min(2.0, timeout_seconds - (now - started))
            if exit_wait <= 0:
                timed_out = True
                failure = "worker sent final report but did not exit before the watchdog deadline"
            else:
                try:
                    process.wait(timeout=exit_wait)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    failure = "worker sent final report but did not exit normally"
            if failure is None:
                reported_exit = candidate_report["outcome"]["exit_code"]
                if process.returncode != reported_exit:
                    failure = (
                        "worker exit code does not match final report "
                        f"(process={process.returncode}, report={reported_exit})"
                    )
                else:
                    final_report = candidate_report
                    received_final = True
                    exit_code = reported_exit

        if final_report is None:
            exit_code = EXIT_TIMEOUT if timed_out else EXIT_RUNTIME
            status = "timeout" if timed_out else "failed"
            final_report = empty_report(
                plan,
                exit_code=exit_code,
                status=status,
                message=failure or "worker failed",
                include_identifiers=bool(plan.get("include_identifiers", False)),
            )
            final_report["diagnostics"].append(
                {"stage": "controller", "code": "worker_timeout" if timed_out else "worker_failure", "message": failure or "worker failed"}
            )
    except (OSError, ProtocolError) as exc:
        exit_code = EXIT_RUNTIME
        final_report = empty_report(
            plan,
            exit_code=exit_code,
            status="failed",
            message=str(exc),
            include_identifiers=bool(plan.get("include_identifiers", False)),
        )
        final_report["diagnostics"].append(
            {"stage": "controller", "code": "isolation_failure", "message": str(exc)}
        )
        trace_records = []
    finally:
        _terminate_worker(process, job)

    cleanup = final_report["cleanup"]
    cleanup["worker_reaped"] = process.poll() is not None
    cleanup["complete"] = all(
        bool(value) for key, value in cleanup.items() if key != "complete"
    )
    worker_trace_complete = received_final and bool(trace_records)
    trace_sequence = max(
        (int(record.get("sequence", 0)) for record in trace_records), default=0
    ) + 1
    trace_records.append(
        {
            "schema_version": 1,
            "record_type": "xvram.pytorch_trace",
            "sequence": trace_sequence,
            "monotonic_ns": time.monotonic_ns(),
            "kind": "verification",
            "region_id": None,
            "allocation_id": None,
            "operation": "trace_complete" if worker_trace_complete else "trace_incomplete",
            "bytes": 0,
            "reason": (
                "worker trace and final received"
                if worker_trace_complete
                else "worker trace or final unavailable"
            ),
        }
    )
    final_report["execution"]["trace_records"] = len(trace_records)
    final_report["execution"]["trace_complete"] = worker_trace_complete
    finalize_proof(final_report)
    if exit_code == EXIT_COMPLETED:
        valid_success, semantic_errors = success_semantics(final_report)
        if not valid_success:
            exit_code = EXIT_RUNTIME
            message = "; ".join(semantic_errors)
            final_report["outcome"].update(
                {"status": "failed", "exit_code": exit_code, "message": message}
            )
            final_report["diagnostics"].append(
                {"stage": "controller", "code": "invalid_success_proof", "message": message}
            )
    if stderr_thread is not None:
        stderr_thread.join(timeout=2)
    if process.stderr is not None:
        process.stderr.close()
    if process.stdout is not None:
        process.stdout.close()
    return ControllerResult(
        report=final_report,
        exit_code=exit_code,
        trace_records=trace_records,
        stderr=bytes(stderr_tail).decode("utf-8", errors="replace"),
    )


def _worker_report(plan: Mapping[str, Any], progress: Callable[[Mapping[str, Any]], None], trace: Callable[[list[dict[str, Any]]], None]) -> dict[str, Any]:
    try:
        from .torch_runtime import run_benchmark_plan
    except (ImportError, AttributeError) as exc:
        raise BenchError(
            EXIT_PREREQUISITE,
            "preflight",
            "runtime_unavailable",
            f"xvram_torch_runtime is unavailable: {exc}",
        ) from exc
    return run_benchmark_plan(dict(plan), progress=progress, trace=trace)


def worker_main(stdin: Optional[BinaryIO] = None, stdout: Optional[BinaryIO] = None) -> int:
    source = stdin or sys.stdin.buffer
    sink = stdout or sys.stdout.buffer
    write_lock = threading.Lock()
    stopped = threading.Event()
    sequence = 0
    operations_retired = 0

    def emit(message_type: MessageType, payload: Mapping[str, Any]) -> None:
        nonlocal sequence
        with write_lock:
            sequence += 1
            write_frame(sink, message_type, {**dict(payload), "sequence": sequence})

    try:
        frame = read_frame(source)
        if frame.message_type is not MessageType.PLAN:
            raise ProtocolError("first XVT1 frame must be plan")
        plan = frame.payload
    except Exception as exc:
        report = empty_report({}, exit_code=EXIT_RUNTIME, status="failed", message=str(exc))
        report["diagnostics"].append(
            {"stage": "protocol", "code": "invalid_plan", "message": str(exc)}
        )
        emit(MessageType.FINAL, {"report": report})
        return EXIT_RUNTIME

    def heartbeat() -> None:
        while not stopped.wait(1.0):
            emit(MessageType.HEARTBEAT, {"monotonic_ns": time.monotonic_ns()})

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    def progress_retired(value: Mapping[str, Any]) -> None:
        nonlocal operations_retired
        operations_retired += 1
        payload = dict(value)
        payload.pop("sequence", None)
        payload["operations_retired"] = operations_retired
        emit(MessageType.PROGRESS, payload)

    try:
        report = _worker_report(
            plan,
            progress=progress_retired,
            trace=lambda records: emit(MessageType.TRACE, {"records": records}),
        )
        exit_code = int(report["outcome"]["exit_code"])
    except BenchError as exc:
        exit_code = exc.exit_code
        status = {
            EXIT_PREREQUISITE: "skipped",
            EXIT_CORRUPTION: "corruption",
            EXIT_PRESSURE: "oom",
            EXIT_TIMEOUT: "timeout",
        }.get(exit_code, "failed")
        report = empty_report(plan, exit_code=exit_code, status=status, message=str(exc), include_identifiers=bool(plan.get("include_identifiers", False)))
        report["diagnostics"].append(
            {"stage": exc.stage, "code": exc.code, "message": str(exc)}
        )
    except BaseException as exc:
        exit_code = EXIT_INTERNAL
        report = empty_report(plan, exit_code=exit_code, status="failed", message=str(exc), include_identifiers=bool(plan.get("include_identifiers", False)))
        report["diagnostics"].append(
            {"stage": "worker", "code": type(exc).__name__, "message": str(exc)}
        )
    finally:
        stopped.set()
        thread.join(timeout=2)
    emit(MessageType.FINAL, {"report": report})
    return exit_code


def _write_output(path: str, content: str) -> None:
    if path == "-":
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content + ("" if content.endswith("\n") else "\n"), encoding="utf-8")


def _text_summary(report: Mapping[str, Any]) -> str:
    outcome = report["outcome"]
    model = report["model"]
    execution = report["execution"]
    cache = report["cache"]
    return "\n".join(
        (
            f"xVRAM PyTorch inference: {outcome['status']} (exit {outcome['exit_code']})",
            f"model: {model['kind']}, layers={model['layers']}, logical={model['logical_bytes']} bytes",
            f"regions: {execution['regions_completed']}, leases: {execution['leases_retired']}",
            f"cache: {cache['hits']} hits, {cache['misses']} misses, {cache['evictions']} evictions",
            f"message: {outcome['message']}",
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="xvram-torch-bench")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", choices=("llama2-like", "operator-smoke"), default="llama2-like")
    parser.add_argument("--model-ratio", type=float, default=1.5)
    parser.add_argument("--layers", type=int, default=31)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--policy", choices=("clock", "lru"), default="clock")
    parser.add_argument("--cache-target", default="auto")
    parser.add_argument("--chunk-size", default="64MiB")
    parser.add_argument("--device-headroom", default="512MiB")
    parser.add_argument("--scratch-cap", default="512MiB")
    parser.add_argument("--prefetch-distance", type=int, choices=range(0, 9), default=2)
    parser.add_argument("--sdpa-backend", choices=("math", "flash_attention"), default="math")
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--trace")
    parser.add_argument("--json", default="-")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--include-identifiers", action="store_true")
    parser.add_argument("--version", action="version", version="xvram-torch-bench 0.1.0-dev")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.worker:
            return worker_main()
        if args.device < 0 or args.layers < 1 or min(args.batch, args.sequence, args.hidden, args.intermediate, args.heads) < 1:
            parser.error("device must be non-negative and model dimensions must be positive")
        if args.hidden % args.heads:
            parser.error("hidden must be divisible by heads")
        if not math.isfinite(args.model_ratio) or args.model_ratio <= 0:
            parser.error("model-ratio must be positive")
        if args.timeout_seconds < 1:
            parser.error("timeout-seconds must be positive")
        if args.seed < 0 or args.seed > 0xFFFFFFFFFFFFFFFF:
            parser.error("seed must fit an unsigned 64-bit value")
        if args.trace == "-" and args.json == "-":
            parser.error("trace and JSON cannot both be written to stdout")
        if args.trace and args.trace != "-" and args.json != "-":
            trace_path = Path(args.trace).expanduser().resolve(strict=False)
            json_path = Path(args.json).expanduser().resolve(strict=False)
            if trace_path == json_path:
                parser.error("trace and JSON must use different output paths")
        plan = _normalize_plan(args)
    except SystemExit as exc:
        return int(exc.code)
    except argparse.ArgumentTypeError as exc:
        parser.print_usage(sys.stderr)
        sys.stderr.write(f"xvram-torch-bench: error: {exc}\n")
        return EXIT_USAGE

    result = run_controller(plan, timeout_seconds=args.timeout_seconds)
    if args.trace:
        try:
            trace_text = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
                for record in result.trace_records
            )
            _write_output(args.trace, trace_text)
        except OSError as exc:
            result.report["execution"]["trace_complete"] = False
            result.report["outcome"].update(
                {"status": "failed", "exit_code": EXIT_OUTPUT, "message": str(exc)}
            )
            result.report["diagnostics"].append(
                {"stage": "output", "code": "trace_io_failure", "message": str(exc)}
            )
            finalize_proof(result.report)
            try:
                _write_output(args.json, dumps(result.report, compact=args.compact_json))
            except OSError:
                pass
            sys.stderr.write(f"trace output I/O failure: {exc}\n")
            return EXIT_OUTPUT
    try:
        _write_output(args.json, dumps(result.report, compact=args.compact_json))
    except OSError as exc:
        sys.stderr.write(f"output I/O failure: {exc}\n")
        return EXIT_OUTPUT
    if not args.no_text:
        sink = sys.stderr if args.json == "-" else sys.stdout
        sink.write(_text_summary(result.report) + "\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
