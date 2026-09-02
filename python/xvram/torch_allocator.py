"""PyTorch ``CUDAPluggableAllocator`` and ``MemPool`` integration.

The module loads the native xVRAM allocator by absolute path and keeps Windows
DLL search-directory cookies alive for as long as the allocator can be used.
Importing :mod:`xvram` itself does not import PyTorch.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


TORCH_ALLOCATOR_ABI_VERSION = 1
XVRAM_TORCH_ALLOCATOR_ENV = "XVRAM_TORCH_ALLOCATOR_LIBRARY"


class XvramTorchError(RuntimeError):
    """Base error raised by the Python adapter."""


class TorchUnavailableError(XvramTorchError):
    """PyTorch or its CUDA allocator API is unavailable."""


class NativeAllocatorError(XvramTorchError):
    """The native allocator rejected an ABI call."""

    def __init__(self, operation: str, status: int, detail: str = "") -> None:
        message = "native allocator {} failed with status {}".format(operation, status)
        if detail:
            message += ": " + detail
        super().__init__(message)
        self.operation = operation
        self.status = int(status)


class _NativeStatsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("allocation_calls", ctypes.c_uint64),
        ("free_calls", ctypes.c_uint64),
        ("allocation_failures", ctypes.c_uint64),
        ("free_failures", ctypes.c_uint64),
        ("requested_bytes_total", ctypes.c_uint64),
        ("requested_bytes_current", ctypes.c_uint64),
        ("requested_bytes_peak", ctypes.c_uint64),
        ("mapped_bytes_total", ctypes.c_uint64),
        ("mapped_bytes_current", ctypes.c_uint64),
        ("mapped_bytes_peak", ctypes.c_uint64),
        ("active_segments", ctypes.c_uint64),
        ("active_segments_peak", ctypes.c_uint64),
        ("reservations", ctypes.c_uint64),
        ("handles_created", ctypes.c_uint64),
        ("maps", ctypes.c_uint64),
        ("set_access_calls", ctypes.c_uint64),
        ("event_boundaries", ctypes.c_uint64),
        ("unmaps", ctypes.c_uint64),
        ("handle_releases", ctypes.c_uint64),
        ("reservation_frees", ctypes.c_uint64),
        ("capture_rejections", ctypes.c_uint64),
        ("context_mismatches", ctypes.c_uint64),
        ("size_mismatches", ctypes.c_uint64),
        ("stream_mismatches", ctypes.c_uint64),
        ("oom_failures", ctypes.c_uint64),
        ("quarantined_segments", ctypes.c_uint64),
        ("quarantined_mapped_bytes", ctypes.c_uint64),
        ("unsafe_unmaps", ctypes.c_uint64),
        ("last_native_error", ctypes.c_int64),
        ("last_status", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _NativeErrorV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("native_code", ctypes.c_int64),
        ("stage", ctypes.c_char * 32),
        ("operation", ctypes.c_char * 64),
        ("message", ctypes.c_char * 256),
    ]


@dataclass(frozen=True)
class NativeAllocatorStats:
    """Read-only snapshot of the allocator's v1 counters."""

    struct_size: int
    abi_version: int
    allocation_calls: int
    free_calls: int
    allocation_failures: int
    free_failures: int
    requested_bytes_total: int
    requested_bytes_current: int
    requested_bytes_peak: int
    mapped_bytes_total: int
    mapped_bytes_current: int
    mapped_bytes_peak: int
    active_segments: int
    active_segments_peak: int
    reservations: int
    handles_created: int
    maps: int
    set_access_calls: int
    event_boundaries: int
    unmaps: int
    handle_releases: int
    reservation_frees: int
    capture_rejections: int
    context_mismatches: int
    size_mismatches: int
    stream_mismatches: int
    oom_failures: int
    quarantined_segments: int
    quarantined_mapped_bytes: int
    unsafe_unmaps: int
    last_native_error: int
    last_status: int

    def to_dict(self) -> Dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class NativeAllocatorErrorInfo:
    """Last diagnostic retained by the native callback boundary."""

    status: int
    native_code: int
    stage: str
    operation: str
    message: str


class XvramTorchAllocator:
    """Own a loaded native plugin and its PyTorch allocator object."""

    def __init__(
        self,
        library_path: Union[str, os.PathLike[str]],
        *,
        _torch_module: Optional[Any] = None,
    ) -> None:
        path = _resolve_library_path(library_path)
        torch_module = _torch_module if _torch_module is not None else _import_torch()
        allocator_type = _cuda_pluggable_allocator_type(torch_module)

        dll_directories = _open_windows_dll_directories(path, torch_module)
        try:
            native = ctypes.CDLL(str(path))
            _configure_native_stats(native)
            allocator = allocator_type(
                str(path),
                "xvram_torch_alloc",
                "xvram_torch_free",
            )
        except Exception:
            _close_dll_directories(dll_directories)
            raise

        self._library_path = path
        self._torch = torch_module
        self._native = native
        self._allocator = allocator
        self._dll_directories = dll_directories

    @property
    def library_path(self) -> Path:
        return self._library_path

    @property
    def torch_allocator(self) -> Any:
        """The underlying ``torch.cuda.memory.CUDAPluggableAllocator``."""

        return self._allocator

    def create_mem_pool(self) -> "XvramMemPool":
        cuda = getattr(self._torch, "cuda", None)
        mem_pool_type = getattr(cuda, "MemPool", None)
        use_mem_pool = getattr(cuda, "use_mem_pool", None)
        if mem_pool_type is None or not callable(mem_pool_type):
            raise TorchUnavailableError("this PyTorch build does not expose torch.cuda.MemPool")
        if use_mem_pool is None or not callable(use_mem_pool):
            raise TorchUnavailableError("this PyTorch build does not expose torch.cuda.use_mem_pool")

        allocator_handle = self._allocator.allocator()
        raw_pool = mem_pool_type(allocator=allocator_handle)
        return XvramMemPool(owner=self, raw_pool=raw_pool)

    def get_stats(self) -> NativeAllocatorStats:
        native_stats = _NativeStatsV1()
        native_stats.struct_size = ctypes.sizeof(_NativeStatsV1)
        native_stats.abi_version = TORCH_ALLOCATOR_ABI_VERSION
        status = int(
            self._native.xvram_torch_get_stats(
                ctypes.byref(native_stats),
                ctypes.sizeof(_NativeStatsV1),
            )
        )
        if status != 0:
            raise NativeAllocatorError("get_stats", status, self._error_detail())
        if native_stats.abi_version != TORCH_ALLOCATOR_ABI_VERSION:
            raise NativeAllocatorError(
                "get_stats",
                status,
                "unsupported ABI version {}".format(native_stats.abi_version),
            )
        if native_stats.struct_size < _minimum_stats_size():
            raise NativeAllocatorError(
                "get_stats",
                status,
                "native stats prefix is {} bytes; at least {} required".format(
                    native_stats.struct_size,
                    _minimum_stats_size(),
                ),
            )
        return _stats_snapshot(native_stats)

    def get_last_error(self) -> NativeAllocatorErrorInfo:
        native_error = _NativeErrorV1()
        native_error.struct_size = ctypes.sizeof(_NativeErrorV1)
        status = int(
            self._native.xvram_torch_get_last_error(
                ctypes.byref(native_error),
                ctypes.sizeof(_NativeErrorV1),
            )
        )
        if status != 0:
            raise NativeAllocatorError("get_last_error", status)
        return NativeAllocatorErrorInfo(
            status=int(native_error.status),
            native_code=int(native_error.native_code),
            stage=_decode_native_text(native_error.stage),
            operation=_decode_native_text(native_error.operation),
            message=_decode_native_text(native_error.message),
        )

    def _error_detail(self) -> str:
        try:
            error = self.get_last_error()
        except NativeAllocatorError:
            return ""
        return ": ".join(part for part in (error.stage, error.operation, error.message) if part)

    def reset_stats(self) -> None:
        status = int(self._native.xvram_torch_reset_stats())
        if status != 0:
            raise NativeAllocatorError("reset_stats", status, self._error_detail())


class XvramMemPool(AbstractContextManager["XvramMemPool"]):
    """A context-manager wrapper that retains its allocator/plugin owner."""

    def __init__(self, *, owner: XvramTorchAllocator, raw_pool: Any) -> None:
        self._owner = owner
        self._raw_pool = raw_pool
        self._contexts = threading.local()

    @property
    def allocator(self) -> XvramTorchAllocator:
        return self._owner

    @property
    def raw_pool(self) -> Any:
        return self._raw_pool

    def use(self) -> AbstractContextManager[Any]:
        """Create a fresh ``torch.cuda.use_mem_pool`` context manager."""

        return self._owner._torch.cuda.use_mem_pool(self._raw_pool)

    def get_stats(self) -> NativeAllocatorStats:
        return self._owner.get_stats()

    def reset_stats(self) -> None:
        self._owner.reset_stats()

    def __enter__(self) -> "XvramMemPool":
        context = self.use()
        stack = getattr(self._contexts, "stack", None)
        if stack is None:
            stack = []
            self._contexts.stack = stack
        stack.append(context)
        try:
            context.__enter__()
        except BaseException:
            stack.pop()
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Optional[bool]:
        stack = getattr(self._contexts, "stack", None)
        if not stack:
            raise RuntimeError("xVRAM MemPool context exit without matching entry")
        context = stack[-1]
        result = context.__exit__(exc_type, exc_value, traceback)
        stack.pop()
        return result


def load(
    library_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> XvramTorchAllocator:
    """Load the native allocator and return its long-lived Python owner.

    When ``library_path`` is omitted, ``XVRAM_TORCH_ALLOCATOR_LIBRARY`` must
    contain the path.  The path is resolved before it is passed to PyTorch.
    """

    if library_path is None:
        configured = os.environ.get(XVRAM_TORCH_ALLOCATOR_ENV)
        if not configured:
            raise ValueError(
                "library_path is required unless {} is set".format(XVRAM_TORCH_ALLOCATOR_ENV)
            )
        library_path = configured
    return XvramTorchAllocator(library_path)


def create_mem_pool(
    library_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> XvramMemPool:
    """Load xVRAM and create a PyTorch MemPool in one call."""

    return load(library_path).create_mem_pool()


def _resolve_library_path(path_value: Union[str, os.PathLike[str]]) -> Path:
    try:
        path = Path(os.fspath(path_value)).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, TypeError) as error:
        raise ValueError("native allocator library does not exist: {!r}".format(path_value)) from error
    if not path.is_file():
        raise ValueError("native allocator path is not a file: {}".format(path))
    if not path.is_absolute():
        raise ValueError("native allocator path did not resolve to an absolute path")
    return path


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise TorchUnavailableError("a CUDA-enabled PyTorch installation is required") from error


def _cuda_pluggable_allocator_type(torch_module: Any) -> Any:
    cuda = getattr(torch_module, "cuda", None)
    memory = getattr(cuda, "memory", None)
    allocator_type = getattr(memory, "CUDAPluggableAllocator", None)
    if allocator_type is None or not callable(allocator_type):
        raise TorchUnavailableError(
            "this PyTorch build does not expose torch.cuda.memory.CUDAPluggableAllocator"
        )
    return allocator_type


def _open_windows_dll_directories(path: Path, torch_module: Any) -> Tuple[Any, ...]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return ()

    candidates = [path.parent]
    torch_file = getattr(torch_module, "__file__", None)
    if torch_file:
        candidates.append(Path(torch_file).resolve().parent / "lib")

    handles: List[Any] = []
    seen = set()
    try:
        for candidate in candidates:
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            if key in seen or not resolved.is_dir():
                continue
            handles.append(os.add_dll_directory(str(resolved)))
            seen.add(key)
    except Exception:
        _close_dll_directories(tuple(handles))
        raise
    return tuple(handles)


def _close_dll_directories(handles: Tuple[Any, ...]) -> None:
    for handle in reversed(handles):
        close = getattr(handle, "close", None)
        if callable(close):
            close()


def _configure_native_stats(native: Any) -> None:
    try:
        get_stats = native.xvram_torch_get_stats
        reset_stats = native.xvram_torch_reset_stats
        get_last_error = native.xvram_torch_get_last_error
    except AttributeError as error:
        raise NativeAllocatorError("load", 1, "required stats export is missing") from error

    get_stats.argtypes = [ctypes.POINTER(_NativeStatsV1), ctypes.c_size_t]
    get_stats.restype = ctypes.c_uint32
    reset_stats.argtypes = []
    reset_stats.restype = ctypes.c_uint32
    get_last_error.argtypes = [ctypes.POINTER(_NativeErrorV1), ctypes.c_size_t]
    get_last_error.restype = ctypes.c_uint32


def _minimum_stats_size() -> int:
    return _NativeStatsV1.last_status.offset + ctypes.sizeof(ctypes.c_uint32)


def _stats_snapshot(stats: _NativeStatsV1) -> NativeAllocatorStats:
    fields = {
        name: int(getattr(stats, name))
        for name, _ctype in _NativeStatsV1._fields_
        if name not in ("reserved0", "reserved")
    }
    return NativeAllocatorStats(**fields)


def _decode_native_text(value: bytes) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")
