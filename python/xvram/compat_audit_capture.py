"""Process isolation for the observation-only compatibility audit.

The bootstrap waits until its parent has assigned the Windows Job Object.  Only
then may it start the unchanged application; on POSIX both share a process group.
No CUDA library is imported by this module.
"""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from .compat_audit_platform import (PlatformObservationUnavailable, drain_windows_job,
                                    loaded_modules, sample_device, sample_process)

MAX_CONTROL_BYTES = 1024 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
_ADDRESS = re.compile(r"\b0[xX][0-9a-fA-F]+\b")
_BARE_POINTER = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{16}(?![0-9A-Za-z])")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def sanitize_log(text: str) -> str:
    return _BARE_POINTER.sub("[native-value-redacted]", _ADDRESS.sub("[native-value-redacted]", _ANSI.sub("", text)))


def parse_timings(stderr: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "load_ms": None, "prefill_ms": None, "prefill_tokens": None,
        "prefill_tokens_per_second": None, "decode_ms": None,
        "decode_tokens": None, "decode_tokens_per_second": None,
    }
    load = re.search(r"load time\s*=\s*([\d.]+)\s*ms", stderr)
    if load:
        result["load_ms"] = float(load.group(1))
    for label, pattern in (("prefill", r"prompt eval time"), ("decode", r"(?<!prompt )eval time")):
        match = re.search(pattern + r"\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs).*?([\d.]+)\s*tokens per second", stderr)
        if match:
            result[f"{label}_ms"] = float(match.group(1))
            result[f"{label}_tokens"] = int(match.group(2))
            result[f"{label}_tokens_per_second"] = float(match.group(3))
    return result


def _run_child(plan: Mapping[str, Any]) -> dict[str, Any]:
    command = plan["command"]
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("invalid child command")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd=plan.get("cwd"))
    logs: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    counts = {"stdout": 0, "stderr": 0}
    first_stdout: list[float] = []
    errors: list[str] = []

    def read_pipe(name: str, pipe: Any) -> None:
        try:
            while True:
                chunk = os.read(pipe.fileno(), 4096)
                if not chunk:
                    break
                if name == "stdout" and not first_stdout:
                    first_stdout.append((time.monotonic() - started) * 1000)
                counts[name] += len(chunk)
                remaining = MAX_LOG_BYTES - len(logs[name])
                logs[name].extend(chunk[:max(0, remaining)])
        except (OSError, ValueError):
            errors.append(f"{name}_read_failure")

    readers = [threading.Thread(target=read_pipe, args=(name, pipe), daemon=True)
               for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr))]
    for reader in readers:
        reader.start()
    memory: dict[str, int | None] = {"peak_rss_bytes": None, "peak_private_bytes": None}
    modules: set[Path] = set()
    module_samples = 0
    module_failures = 0
    next_module_sample = 0.0
    device_samples: list[dict[str, Any]] = []
    sampler_stop = threading.Event()

    def observe_device() -> None:
        while not sampler_stop.is_set() and len(device_samples) < 86400:
            device_samples.append(sample_device(timeout_seconds=2))
            sampler_stop.wait(1)

    device_thread = threading.Thread(target=observe_device, daemon=True)
    if plan.get("sample_gpu", False):
        device_thread.start()
    while process.poll() is None:
        sample = sample_process(process)
        for target, source in (("peak_rss_bytes", "peak_rss_bytes"), ("peak_private_bytes", "private_bytes")):
            value = sample[source]
            if value is not None:
                memory[target] = max(memory[target] or 0, value)
        if time.monotonic() >= next_module_sample:
            try:
                observed = loaded_modules(process)
                merged_modules = modules.union(observed)
                if len(merged_modules) > 4096:
                    if "module_history_limit" not in errors:
                        errors.append("module_history_limit")
                else:
                    modules = merged_modules
                module_samples += 1
            except PlatformObservationUnavailable:
                module_failures += 1
            next_module_sample = time.monotonic() + 0.25
        time.sleep(0.05)
    elapsed_ms = (time.monotonic() - started) * 1000
    sampler_stop.set()
    if device_thread.is_alive():
        device_thread.join(timeout=3)
    for reader in readers:
        reader.join(timeout=5)
    if any(reader.is_alive() for reader in readers):
        errors.append("pipe_not_closed_by_descendant")
    text = {name: sanitize_log(bytes(data).decode("utf-8", errors="replace")) for name, data in logs.items()}
    for name, data in text.items():
        (output / f"{name}.txt").write_text(data, encoding="utf-8")
    truncated = any(value > MAX_LOG_BYTES for value in counts.values())
    module_evidence = []
    for path in sorted(modules, key=lambda item: str(item).casefold()):
        basename = path.name.lower().removeprefix("lib")
        if not basename.startswith(("llama", "ggml", "cublas", "cudart", "nvcuda", "cupti", "nvapi", "nvwgf")):
            continue
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while data := stream.read(4 * 1024 * 1024):
                    digest.update(data)
            module_evidence.append({"name": path.name, "sha256": digest.hexdigest()})
        except OSError:
            errors.append("observed_module_hash_unavailable")
    good_device_samples = [sample for sample in device_samples if sample.get("available")]
    gpu = {"scope": "whole_device_not_process", "samples": len(good_device_samples),
           "peak_used_bytes": max((sample["used_bytes"] for sample in good_device_samples), default=None),
           "min_free_bytes": min((sample["free_bytes"] for sample in good_device_samples), default=None)}
    return {
        "exit_code": process.returncode, "timed_out": False, "controller_reaped": True,
        "elapsed_ms": elapsed_ms,
        "first_stdout_ms": first_stdout[0] if first_stdout else None,
        "ttft_ms": None, "ttft_status": "not_identifiable_from_cli_output",
        "timings": parse_timings(text["stderr"]), "memory": {**memory, "scope": "direct_command_process"},
        "gpu": gpu, "loaded_modules": module_evidence,
        "module_snapshot_count": module_samples, "module_snapshot_failures": module_failures,
        "module_coverage_complete": False,
        "stdout_sha256": hashlib.sha256(text["stdout"].encode()).hexdigest(),
        "output_truncated": truncated, "errors": errors,
    }


def run_process(command: Sequence[str], *, output_dir: Path, timeout_seconds: float = 900,
                environment: Mapping[str, str] | None = None, cwd: Path | None = None,
                sample_gpu: bool = False) -> dict[str, Any]:
    """Run only our new child tree; never attach to or terminate another process."""
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86400:
        raise ValueError("timeout must be finite and between 0 and 86400 seconds")
    plan = json.dumps({"command": list(command), "output_dir": str(output_dir.resolve()),
                       "cwd": str(cwd.resolve()) if cwd else None, "sample_gpu": sample_gpu}).encode()
    if len(plan) >= MAX_CONTROL_BYTES:
        raise ValueError("child command exceeds control limit")
    env = dict(os.environ if environment is None else environment)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    worker = subprocess.Popen([sys.executable, "-m", "xvram.compat_audit_capture", "--child"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=env, start_new_session=os.name != "nt")
    job = None
    started = time.monotonic()
    result: dict[str, Any] | None = None
    control = bytearray()
    diagnostic = bytearray()
    control_ready = threading.Event()
    control_errors: list[str] = []
    reader_threads: list[threading.Thread] = []

    def read_control() -> None:
        # os.read avoids holding a BufferedReader lock during final pipe closure.
        try:
            while len(control) <= MAX_CONTROL_BYTES:
                chunk = os.read(worker.stdout.fileno(), 4096)
                if not chunk:
                    control_errors.append("bootstrap_control_eof")
                    break
                control.extend(chunk)
                if b"\n" in chunk:
                    break
            if len(control) > MAX_CONTROL_BYTES:
                control_errors.append("bootstrap_control_limit")
        except (OSError, ValueError):
            control_errors.append("bootstrap_control_read_failure")
        finally:
            control_ready.set()

    def read_diagnostic() -> None:
        try:
            while chunk := os.read(worker.stderr.fileno(), 4096):
                diagnostic.extend(chunk[:max(0, 2048 - len(diagnostic))])
        except (OSError, ValueError):
            pass

    def send_plan() -> None:
        try:
            # Keep stdin open. The bootstrap remains our owned group leader until
            # teardown, even after its audited command has exited.
            remaining = memoryview(plan + b"\n")
            while remaining:
                written = os.write(worker.stdin.fileno(), remaining)
                if written <= 0:
                    raise OSError("control write failed")
                remaining = remaining[written:]
        except (OSError, ValueError):
            control_errors.append("bootstrap_control_write_failure")
            control_ready.set()

    try:
        if os.name == "nt":
            # Reuse the existing tested containment primitive without modifying it.
            from .torch_bench import _WindowsJob
            job = _WindowsJob(worker)
        reader_threads = [threading.Thread(target=target, daemon=True)
                          for target in (read_control, read_diagnostic, send_plan)]
        for thread in reader_threads:
            thread.start()
        if not control_ready.wait(max(0, timeout_seconds - (time.monotonic() - started))):
            result = {"exit_code": 26, "timed_out": True, "errors": ["deadline_exceeded"]}
        else:
            try:
                if control_errors or not control.endswith(b"\n") or len(control) > MAX_CONTROL_BYTES:
                    raise ValueError("invalid child framing")
                parsed = json.loads(control)
                if (not isinstance(parsed, dict) or type(parsed.get("exit_code")) is not int
                        or not isinstance(parsed.get("errors"), list)
                        or not all(isinstance(item, str) for item in parsed["errors"])):
                    raise ValueError("invalid child result")
                result = parsed
            except (ValueError, UnicodeError):
                result = {"exit_code": 27, "timed_out": False,
                          "errors": ["invalid_child_result"]}
    except (OSError, RuntimeError) as error:
        result = {"exit_code": 27, "timed_out": False,
                  "errors": ["bootstrap_failed", type(error).__name__]}
    finally:
        # Do not poll/reap a POSIX group leader before signaling the owned group:
        # retaining its PID prevents stale PGID reuse after a fast bootstrap exit.
        drained = False
        try:
            drained = _terminate(worker, job)
        finally:
            if job is not None:
                with contextlib.suppress(OSError, RuntimeError):
                    job.close()
            for stream in (worker.stdin, worker.stdout, worker.stderr):
                if stream is not None:
                    with contextlib.suppress(OSError, ValueError):
                        stream.close()
            for thread in reader_threads:
                if thread.ident is not None:
                    thread.join(timeout=1)
        if result is not None:
            result["controller_reaped"] = worker.returncode is not None
            result["process_tree_drained"] = drained
            result["containment_scope"] = "owned_windows_job" if os.name == "nt" else "owned_process_group"
            result.setdefault("elapsed_ms", (time.monotonic() - started) * 1000)
            if not drained or worker.returncode is None:
                result["errors"].append("process_tree_not_drained")
                if not result.get("timed_out"):
                    result["exit_code"] = 27
    assert result is not None
    return result


def _posix_group_has_live_members(group_id: int) -> bool | None:
    """Read-only Linux verification while the group leader remains unreaped.

    Zombies cannot execute; their eventual reaping may belong to init. Other
    platforms or unreadable membership evidence remain unknown, never success.
    A process that deliberately leaves the group is outside this containment
    scope; this is not a cgroup/subreaper implementation.
    """
    try:
        entries = list(itertools.islice(Path("/proc").iterdir(), 131073))
        if len(entries) > 131072:
            return None
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                text = (entry / "stat").read_text(encoding="utf-8")
            except FileNotFoundError:
                continue  # A concurrent unrelated process exited.
            except (OSError, UnicodeError):
                return None
            tail = text[text.rfind(")") + 2:].split()
            if len(tail) < 4:
                return None
            if int(tail[2]) == group_id and tail[0] not in ("Z", "X"):
                return True
        return False
    except (OSError, ValueError):
        return None


def _terminate(worker: subprocess.Popen[bytes], job: Any) -> bool:
    drained = False
    if job is not None:
        try:
            drained = drain_windows_job(job)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            drained = False
        finally:
            # Close the kill-on-close job even when query/terminate/wait fails.
            # That fallback contains children, but cannot prove a successful drain.
            try:
                job.close()
            except (OSError, RuntimeError):
                drained = False
    elif os.name != "nt":
        if worker.returncode is not None:
            # The numeric process-group identity may already have been reused.
            return False
        try:
            os.killpg(worker.pid, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                live = _posix_group_has_live_members(worker.pid)
                if live is not True:
                    drained = live is False
                    break
                time.sleep(0.01)
        except ProcessLookupError:
            # An owned but unreaped leader still reserves its PID, so this is not
            # a stale-ID success. Verify rather than claiming an empty group.
            drained = _posix_group_has_live_members(worker.pid) is False
        except OSError:
            drained = False
    else:
        # Assignment failed before the bootstrap received its launch plan.
        with contextlib.suppress(OSError):
            worker.kill()
    try:
        worker.wait(timeout=10)
        if job is None and os.name == "nt":
            drained = True
    except (OSError, subprocess.TimeoutExpired):
        drained = False
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            worker.kill()
            worker.wait(timeout=2)
    return drained


def main() -> int:
    if sys.argv[1:] != ["--child"]:
        return 64
    raw = sys.stdin.buffer.readline(MAX_CONTROL_BYTES + 1)
    if not raw.endswith(b"\n") or len(raw) > MAX_CONTROL_BYTES:
        return 64
    try:
        result = _run_child(json.loads(raw))
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) + 1 > MAX_CONTROL_BYTES:
            return 27
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()
        # Keep the containment leader alive and unreaped until the controller has
        # consumed its result and tears down the owned group/job.
        sys.stdin.buffer.read(1)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.stderr.write(type(error).__name__)
        return 27


if __name__ == "__main__":
    raise SystemExit(main())
