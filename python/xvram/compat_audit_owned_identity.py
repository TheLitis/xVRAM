"""Private post-reap process identity for ETW joins, never public report data.

The capture bootstrap queries its original Popen handle. It never reopens a
process by PID, searches by executable name, or changes the child. Only an
explicit parent-owned named mapping enables this side channel. The ordinary
capture report/schema is unchanged. This is observation, not a security sandbox.
"""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import mmap
import os
import secrets
import struct
import zlib

ENVIRONMENT = "XVRAM_AUDIT_OWNED_IDENTITY"
MAGIC = 0x3144494F415658
FIELDS = ("magic version bytes nonce state pid creation_filetime exit_filetime "
          "frequency prepared_qpc start_before_qpc start_after_qpc "
          "finish_before_qpc finish_after_qpc exit_code checksum").split()
FORMAT = struct.Struct("<"+"Q"*len(FIELDS))
_active = None


@dataclass(frozen=True, repr=False)
class PrivateIdentity:
    native_pid: int
    creation_filetime: int
    exit_filetime: int
    qpc_frequency: int
    prepared_qpc: int
    start_before_qpc: int
    start_after_qpc: int
    finish_before_qpc: int
    finish_after_qpc: int
    reaped_qpc: int

    def __repr__(self):
        return "<PrivateIdentity: process-local ETW correlation data, redacted>"


def _pack(row):
    values = [row[key] for key in FIELDS]
    values[-1] = zlib.crc32(FORMAT.pack(*values)[:-8])
    return FORMAT.pack(*values)


def _unpack(data):
    if len(data) != FORMAT.size:
        raise ValueError("owned_identity_size")
    row = dict(zip(FIELDS, FORMAT.unpack(data)))
    if (row["magic"], row["version"], row["bytes"]) != (MAGIC, 1, FORMAT.size):
        raise ValueError("owned_identity_header")
    if row["checksum"] != zlib.crc32(data[:-8]):
        raise ValueError("owned_identity_checksum")
    return row


def decode(data, capture, *, nonce, frequency, reaped_qpc):
    """Requires a clean bounded-controller reap before exposing private fields."""
    if (capture.get("controller_reaped") is not True or capture.get("process_tree_drained") is not True or
            capture.get("timed_out") is not False or type(capture.get("exit_code")) is not int or
            capture["exit_code"] != 0 or capture.get("errors") != []):
        raise ValueError("owned_identity_worker_not_cleanly_reaped")
    row = _unpack(data)
    if type(nonce) is not int or not 0 < nonce < 2**64 or row["nonce"] != nonce:
        raise ValueError("owned_identity_mapping_generation")
    if type(frequency) is not int or not 0 < frequency < 2**63 or row["frequency"] != frequency:
        raise ValueError("owned_identity_clock_frequency")
    if type(reaped_qpc) is not int or not 0 < reaped_qpc < 2**63:
        raise ValueError("owned_identity_reap_clock")
    if row["state"] != 2 or row["exit_code"] != 0:
        raise ValueError("owned_identity_unfinished_or_failed")
    if not 0 < row["pid"] < 2**32 or not 0 < row["creation_filetime"] <= row["exit_filetime"] < 2**63:
        raise ValueError("owned_identity_process_times")
    if not (0 < row["prepared_qpc"] <= row["start_before_qpc"] <= row["start_after_qpc"] <=
            row["finish_before_qpc"] <= row["finish_after_qpc"] <= reaped_qpc):
        raise ValueError("owned_identity_clock_order")
    return PrivateIdentity(row["pid"], row["creation_filetime"], row["exit_filetime"], frequency,
                           row["prepared_qpc"], row["start_before_qpc"], row["start_after_qpc"],
                           row["finish_before_qpc"], row["finish_after_qpc"], reaped_qpc)


def _kernel():
    if os.name != "nt":
        raise OSError("owned_identity_windows_only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    kernel.QueryPerformanceCounter.restype = wintypes.BOOL
    kernel.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    kernel.QueryPerformanceFrequency.restype = wintypes.BOOL
    kernel.GetProcessId.argtypes = [wintypes.HANDLE]
    kernel.GetProcessId.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    return kernel


def _clock(kernel, frequency=False):
    value = ctypes.c_longlong()
    function = kernel.QueryPerformanceFrequency if frequency else kernel.QueryPerformanceCounter
    if not function(ctypes.byref(value)) or value.value <= 0:
        raise OSError("owned_identity_clock_unavailable")
    return value.value


def _times(kernel, process):
    # CPython retains this original owned handle after poll/wait. No OpenProcess
    # or PID lookup is permitted; identity is checked against the handle itself.
    handle = getattr(process, "_handle", None)
    if handle is None:
        raise ValueError("owned_identity_original_handle_missing")
    pid = kernel.GetProcessId(handle)
    if not pid or type(process.pid) is not int or pid != process.pid:
        raise ValueError("owned_identity_original_handle_mismatch")
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
        raise OSError("owned_identity_process_times_unavailable")
    combined = [value.dwLowDateTime | value.dwHighDateTime << 32 for value in times]
    return pid, combined[0], combined[1]


class Ledger:
    """Parent mapping lifetime extends beyond bounded bootstrap/worker reap."""
    def __init__(self):
        kernel = _kernel()
        self.nonce = secrets.randbits(64) or 1
        self.frequency = _clock(kernel, frequency=True)
        self.name = "Local\\xvram-owned-"+secrets.token_hex(24)
        self.mapping = mmap.mmap(-1, FORMAT.size, tagname=self.name, access=mmap.ACCESS_WRITE)
        row = dict.fromkeys(FIELDS, 0)
        row.update(magic=MAGIC, version=1, bytes=FORMAT.size, nonce=self.nonce,
                   frequency=self.frequency, prepared_qpc=_clock(kernel))
        self.mapping[:] = _pack(row)

    def environment(self):
        return {ENVIRONMENT: self.name}

    def snapshot_after_reap(self, capture):
        return decode(self.mapping[:], capture, nonce=self.nonce, frequency=self.frequency,
                      reaped_qpc=_clock(_kernel()))

    def close(self):
        self.mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def start(process):
    """Bootstrap opt-in hook immediately after Popen. No-op without environment."""
    global _active
    name = os.environ.get(ENVIRONMENT)
    if name is None:
        return
    prefix = "Local\\xvram-owned-"
    if (_active is not None or not name.startswith(prefix) or len(name) != len(prefix)+48 or
            any(char not in "0123456789abcdef" for char in name[len(prefix):])):
        raise ValueError("owned_identity_mapping_name_or_reinitialize")
    kernel = _kernel()
    mapping = mmap.mmap(-1, FORMAT.size, tagname=name, access=mmap.ACCESS_WRITE)
    try:
        row = _unpack(mapping[:])
        if row["state"] != 0 or not row["nonce"] or row["frequency"] != _clock(kernel, frequency=True):
            raise ValueError("owned_identity_prepared_mapping")
        if any(row[key] for key in ("pid", "creation_filetime", "exit_filetime", "start_before_qpc",
                                   "start_after_qpc", "finish_before_qpc", "finish_after_qpc", "exit_code")):
            raise ValueError("owned_identity_nonempty_mapping")
        before = _clock(kernel)
        pid, creation, _ = _times(kernel, process)
        after = _clock(kernel)
        if not 0 < row["prepared_qpc"] <= before <= after or not creation:
            raise ValueError("owned_identity_start_clock")
        row.update(state=1, pid=pid, creation_filetime=creation, start_before_qpc=before, start_after_qpc=after)
        mapping[:] = _pack(row)
        _active = (process, mapping, _unpack(mapping[:]))
    except BaseException:
        mapping.close()
        raise


def finish(process):
    """Bootstrap hook only after the direct child has actually terminated."""
    global _active
    if os.environ.get(ENVIRONMENT) is None:
        return
    if _active is None or _active[0] is not process:
        raise ValueError("owned_identity_not_started_or_wrong_child")
    _, mapping, row = _active
    if row["state"] != 1:
        raise ValueError("owned_identity_duplicate_finish")
    kernel = _kernel()
    if type(process.returncode) is not int or kernel.WaitForSingleObject(process._handle, 0) != 0:
        raise ValueError("owned_identity_child_still_running")
    before = _clock(kernel)
    pid, creation, exit_time = _times(kernel, process)
    code = wintypes.DWORD()
    if not kernel.GetExitCodeProcess(process._handle, ctypes.byref(code)):
        raise OSError("owned_identity_exit_code_unavailable")
    after = _clock(kernel)
    if pid != row["pid"] or creation != row["creation_filetime"] or not creation <= exit_time or not exit_time:
        raise ValueError("owned_identity_child_generation_changed")
    if code.value != process.returncode or not row["start_after_qpc"] <= before <= after:
        raise ValueError("owned_identity_exit_or_clock_mismatch")
    if _unpack(mapping[:]) != row:
        raise ValueError("owned_identity_mapping_changed")
    row.update(state=2, exit_filetime=exit_time, finish_before_qpc=before, finish_after_qpc=after, exit_code=code.value)
    mapping[:] = _pack(row)
    mapping.close()
