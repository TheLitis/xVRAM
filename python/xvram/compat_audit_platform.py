"""Bounded OS observations of a controller-owned child; never load CUDA.

Module paths are transient internal evidence. Callers must hash/select them locally
and serialize only approved basenames/digests, never paths, pointers, or OS handles.
Snapshots are observations, not proof that every briefly loaded module was observed.
"""
from __future__ import annotations

import csv
import ctypes
import functools
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

MAX_MODULES = 2048
MAX_PROC_MAP_BYTES = 8 * 1024 * 1024
MAX_MODULE_PATH_CHARS = 32768
_UINT64_MAX = (1 << 64) - 1


class PlatformObservationUnavailable(RuntimeError):
    """A sanitized observation failure; no empty result should imply completeness."""


@functools.lru_cache(maxsize=1)
def _kernel32() -> Any:
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _process_handle(process: subprocess.Popen[bytes]) -> ctypes.c_void_p:
    # Only accept the already-owned Popen object, not a PID or arbitrary handle API.
    try:
        return ctypes.c_void_p(int(process._handle))
    except (AttributeError, TypeError, ValueError) as error:
        raise PlatformObservationUnavailable("owned_process_handle_unavailable") from error


def _windows_loaded_modules(process: subprocess.Popen[bytes]) -> list[Path]:
    kernel = _kernel32()
    enumerate_modules = kernel.K32EnumProcessModulesEx
    enumerate_modules.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                                  ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    enumerate_modules.restype = ctypes.c_int
    module_filename = kernel.K32GetModuleFileNameExW
    module_filename.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    module_filename.restype = ctypes.c_uint32
    handle = _process_handle(process)
    modules = (ctypes.c_void_p * MAX_MODULES)()
    required_bytes = ctypes.c_uint32()
    if not enumerate_modules(handle, modules, ctypes.sizeof(modules), ctypes.byref(required_bytes), 3):
        raise PlatformObservationUnavailable("module_enumeration_failed")
    handle_bytes = ctypes.sizeof(ctypes.c_void_p)
    if required_bytes.value == 0:
        raise PlatformObservationUnavailable("module_snapshot_empty")
    if required_bytes.value > ctypes.sizeof(modules) or required_bytes.value % handle_bytes:
        raise PlatformObservationUnavailable("module_snapshot_exceeds_bound")
    paths: set[Path] = set()
    for index in range(required_bytes.value // handle_bytes):
        buffer = ctypes.create_unicode_buffer(MAX_MODULE_PATH_CHARS)
        length = module_filename(handle, modules[index], buffer, len(buffer))
        if length == 0:
            raise PlatformObservationUnavailable("module_path_query_failed_or_snapshot_changed")
        if length >= len(buffer) - 1:
            raise PlatformObservationUnavailable("module_path_truncated")
        path = Path(buffer.value)
        if not path.is_absolute():
            raise PlatformObservationUnavailable("module_path_not_absolute")
        paths.add(path)
    if not paths:
        raise PlatformObservationUnavailable("module_snapshot_empty")
    return sorted(paths, key=lambda path: str(path).casefold())


def _proc_module_paths(text: str) -> list[Path]:
    paths: set[Path] = set()
    for line in text.splitlines():
        columns = line.split(None, 5)
        if len(columns) < 5:
            raise PlatformObservationUnavailable("proc_maps_malformed")
        if len(columns) != 6 or not columns[5].startswith("/"):
            continue
        value = columns[5]
        if value.endswith(" (deleted)"):
            raise PlatformObservationUnavailable("mapped_file_deleted")
        value = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)
        paths.add(Path(value))
        if len(paths) > MAX_MODULES:
            raise PlatformObservationUnavailable("module_snapshot_exceeds_bound")
    if not paths:
        raise PlatformObservationUnavailable("module_snapshot_empty")
    return sorted(paths, key=str)


def loaded_modules(process: subprocess.Popen[bytes]) -> list[Path]:
    """Return one bounded path snapshot; failures are unknown, never verified empty.

    Windows PSAPI snapshots can race loading/unloading. Repeated successful snapshots
    improve observations but do not establish complete dynamic-load coverage. Returned
    module handles are borrowed snapshots and must never be closed by this helper.
    Linux returns file-backed /proc mappings, which are not all executable modules.
    """
    try:
        if os.name == "nt":
            return _windows_loaded_modules(process)
        with Path(f"/proc/{process.pid}/maps").open("rb") as stream:
            raw = stream.read(MAX_PROC_MAP_BYTES + 1)
        if len(raw) > MAX_PROC_MAP_BYTES:
            raise PlatformObservationUnavailable("proc_maps_exceeds_bound")
        return _proc_module_paths(raw.decode("utf-8", errors="strict"))
    except PlatformObservationUnavailable:
        raise
    except (OSError, ValueError, UnicodeError) as error:
        raise PlatformObservationUnavailable("module_snapshot_unavailable") from error


def sample_process(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Sample only the direct command process, not its child tree or GPU allocation."""
    observation: dict[str, Any] = {
        "available": False, "rss_bytes": None, "peak_rss_bytes": None,
        "private_bytes": None, "scope": "direct_command_process", "error": None,
    }
    try:
        if os.name == "nt":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
                    )
                ]
            info = Counters()
            info.cb = ctypes.sizeof(info)
            query = _kernel32().K32GetProcessMemoryInfo
            query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
            query.restype = ctypes.c_int
            if not query(_process_handle(process), ctypes.byref(info), info.cb):
                raise PlatformObservationUnavailable("process_memory_query_failed")
            observation.update(available=True, rss_bytes=int(info.WorkingSetSize),
                               peak_rss_bytes=int(info.PeakWorkingSetSize), private_bytes=int(info.PrivateUsage))
        else:
            with Path(f"/proc/{process.pid}/status").open("r", encoding="utf-8") as stream:
                text = stream.read(65537)
            if len(text) > 65536:
                raise PlatformObservationUnavailable("proc_status_exceeds_bound")
            values = {}
            for line in text.splitlines():
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    fields = value.split()
                    if len(fields) != 2 or fields[1] != "kB" or not fields[0].isdigit():
                        raise PlatformObservationUnavailable("proc_status_malformed")
                    amount = int(fields[0]) * 1024
                    if amount > _UINT64_MAX:
                        raise PlatformObservationUnavailable("proc_status_overflow")
                    values[key] = amount
            if "VmRSS" not in values or "VmHWM" not in values:
                raise PlatformObservationUnavailable("process_memory_query_failed")
            observation.update(available=True, rss_bytes=values["VmRSS"], peak_rss_bytes=values["VmHWM"])
    except (OSError, ValueError, UnicodeError, PlatformObservationUnavailable):
        observation["error"] = "process_memory_unavailable"
    return observation


def _parse_device_csv(text: str, device: int) -> dict[str, Any]:
    records = list(csv.reader(text.strip().splitlines()))
    if len(records) != 1 or len(records[0]) != 5:
        raise ValueError("device_query_shape")
    name, total, used, free, driver = (value.strip() for value in records[0])
    amounts = []
    for value in (total, used, free):
        if not value.isdigit():
            raise ValueError("device_memory_unavailable")
        amount = int(value) * 1024**2
        if amount > _UINT64_MAX:
            raise ValueError("device_memory_overflow")
        amounts.append(amount)
    if not name or not driver or amounts[1] > amounts[0] or amounts[2] > amounts[0]:
        raise ValueError("device_query_inconsistent")
    return {"available": True, "name": name, "total_bytes": amounts[0],
            "used_bytes": amounts[1], "free_bytes": amounts[2], "driver_version": driver,
            "device_ordinal": device, "scope": "whole_device_not_process", "source": "nvidia_smi", "error": None}


def sample_device(device: int = 0, *, executable: str | None = None,
                  timeout_seconds: float = 2) -> dict[str, Any]:
    """Read a whole-device observation; callers rate-limit to at most once/second.

    This is not a CUDA/WDDM admission budget or a per-process GPU-memory measurement.
    nvidia-smi index ordering must be reconciled with the audited device separately.
    """
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a nonnegative ordinal")
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise ValueError("device query timeout must be finite and bounded")
    unavailable = {"available": False, "name": None, "total_bytes": None,
                   "used_bytes": None, "free_bytes": None, "driver_version": None,
                   "device_ordinal": device, "scope": "whole_device_not_process",
                   "source": "nvidia_smi", "error": "device_query_unavailable"}
    selected = executable or shutil.which("nvidia-smi")
    if not selected:
        return unavailable
    try:
        result = subprocess.run(
            [selected, "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits", f"--id={device}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=timeout_seconds, check=True,
        )
        if len(result.stdout) > 65536:
            return unavailable
        return _parse_device_csv(result.stdout, device)
    except (OSError, ValueError, subprocess.SubprocessError):
        return unavailable


class _JobAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in (
        "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime",
    )] + [(name, ctypes.c_uint32) for name in (
        "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses",
    )]


def drain_windows_job(job: Any, terminate: bool = True, timeout_seconds: float = 10) -> bool:
    """Drain only the supplied owned job; leave its handle open for caller cleanup.

    Unlike closing a kill-on-close job alone, a zero ActiveProcesses observation proves
    that its descendants have terminated. Query/termination failures remain unknown.
    """
    if not math.isfinite(timeout_seconds) or not 0 <= timeout_seconds <= 60:
        raise ValueError("job drain timeout must be finite and bounded")
    handle = getattr(job, "_handle", None)
    kernel = getattr(job, "_kernel32", None)
    if not handle or kernel is None:
        return False
    try:
        query = kernel.QueryInformationJobObject
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        query.restype = ctypes.c_int
        stop = kernel.TerminateJobObject
        stop.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        stop.restype = ctypes.c_int
        deadline = time.monotonic() + timeout_seconds
        requested_stop = False
        while True:
            accounting = _JobAccounting()
            if not query(ctypes.c_void_p(handle), 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                return False
            if accounting.ActiveProcesses == 0:
                return True
            if terminate and not requested_stop:
                if not stop(ctypes.c_void_p(handle), 1):
                    return False
                requested_stop = True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))
    except (OSError, AttributeError, TypeError, ValueError):
        return False
