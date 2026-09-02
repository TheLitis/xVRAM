"""Native Phase 4b backend for lease-scoped PyTorch inference.

The module deliberately keeps CUDA identities on the native side.  Python sees
only opaque session/allocation/lease handles and the runtime-owned stream used
to construct a short-lived :class:`torch.cuda.ExternalStream`.  Managed CUDA
tensors are created by ``xvram_internal::_wrap_resolved_v1`` and all references
to them are destroyed before the lease is sealed.

Importing this module does not import PyTorch or load a native library.  Both
operations are delayed until :meth:`NativeInferenceBackend.prepare`, which is
called in the isolated worker process after the controller has finished its
preflight work.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import importlib
import os
import threading
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .torch_planner import AccessMode, InferencePlan, ManagedRange, ManagedValue, ValueKind, ValueReference
from .torch_runtime import (
    BackendContractError,
    ExecutionRequest,
    InferenceBackend,
    InferenceBackendFactory,
    StateProvider,
)


TORCH_RUNTIME_ABI_VERSION = 1
XVRAM_TORCH_RUNTIME_ENV = "XVRAM_TORCH_RUNTIME_LIBRARY"

_STATUS_SUCCESS = 0
_STATUS_VIEWS_LIVE = 5
_STATUS_TIMEOUT = 11

_POLICIES = {"clock": 1, "lru": 2}
_HINTS = {"normal": 1, "hot": 2, "streaming": 3}
_ACCESS_MODES = {
    AccessMode.READ: 1,
    AccessMode.READ_WRITE: 2,
    AccessMode.WRITE_ONLY: 3,
}
_SEAL_SUCCESS = 1
_SEAL_CANCELLED = 2
_SEAL_FAILED_AFTER_SUBMISSION = 3
_LEASE_ARMED = 1
_LEASE_SUBMITTED = 2
_LEASE_COMPLETED = 3
_LEASE_CANCELLED = 4
_LEASE_FAILED = 5
_LEASE_QUARANTINED = 6

_SCALAR_TYPES = {
    "torch.uint8": 0,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.int32": 3,
    "torch.int64": 4,
    "torch.float16": 5,
    "torch.float32": 6,
    "torch.float64": 7,
    "torch.bool": 11,
    "torch.bfloat16": 15,
}

_BASIC_OUT_ADAPTERS = {
    "embedding_out": "embedding",
    "linear_out": "linear",
    "mm_out": "mm",
    "addmm_out": "addmm",
    "bmm_out": "bmm",
    "add_out": "add",
    "mul_out": "mul",
    "neg_out": "neg",
    "cat_out": "cat",
    "silu_out": "silu",
}
_SCRATCH_ADAPTERS = frozenset({"rms_norm_scratch_copy", "sdpa_math_scratch_copy"})
_GEMM_DTYPES = {
    "torch.float16": 1,
    "torch.bfloat16": 2,
    "torch.float32": 3,
    "torch.float64": 4,
}
_GEMM_COMPLETED = 1


class NativeTorchRuntimeError(RuntimeError):
    """A private native runtime call failed or violated its ABI contract."""

    def __init__(
        self,
        operation: str,
        status: int,
        detail: str = "",
        *,
        native_code: int = 0,
    ) -> None:
        message = "native torch runtime {} failed with status {}".format(operation, status)
        if detail:
            message += ": " + detail
        super().__init__(message)
        self.operation = operation
        self.status = int(status)
        self.native_code = int(native_code)


class NativeTorchUnavailableError(NativeTorchRuntimeError):
    """The native bridge, CUDA-enabled PyTorch, or a required API is absent."""


@dataclass(frozen=True)
class NativeRuntimeConfig:
    """Configuration passed verbatim to one native residency session."""

    device: int = 0
    policy: str = "clock"
    chunk_size: Union[int, str] = "64MiB"
    cache_target: Union[int, str] = "auto"
    device_headroom: Union[int, str] = "512MiB"
    scratch_cap: Union[int, str] = "512MiB"
    staging_slots: int = 4
    stall_timeout_ms: int = 5000
    budget_poll_ms: int = 100
    maximum_region_ms: int = 250
    close_timeout_ms: int = 5000
    prefetch_distance: int = 2
    gemm_workspace: Union[int, str] = "4MiB"
    gemm_tile_m: int = 4096
    gemm_tile_n: int = 4096
    gemm_tile_k: int = 1024
    sdpa_backend: str = "math"

    def validate(self) -> "NativeRuntimeConfig":
        if isinstance(self.device, bool) or int(self.device) < 0:
            raise ValueError("device must be a non-negative ordinal")
        if str(self.policy).lower() not in _POLICIES:
            raise ValueError("policy must be 'clock' or 'lru'")
        chunk = _parse_bytes(self.chunk_size, allow_auto=False)
        target = _parse_bytes(self.cache_target, allow_auto=True)
        headroom = _parse_bytes(self.device_headroom, allow_auto=False)
        scratch = _parse_bytes(self.scratch_cap, allow_auto=False)
        workspace = _parse_bytes(self.gemm_workspace, allow_auto=False)
        if chunk <= 0 or headroom <= 0 or scratch <= 0 or target < 0:
            raise ValueError("runtime sizes must be positive (cache target may be auto)")
        if workspace > scratch:
            raise ValueError("gemm_workspace cannot exceed scratch_cap")
        if not 2 <= int(self.staging_slots) <= 8:
            raise ValueError("staging_slots must be in [2, 8]")
        if not 0 <= int(self.prefetch_distance) <= 8:
            raise ValueError("prefetch_distance must be in [0, 8]")
        for name, value in (
            ("gemm_tile_m", self.gemm_tile_m),
            ("gemm_tile_n", self.gemm_tile_n),
            ("gemm_tile_k", self.gemm_tile_k),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError("{} must be positive".format(name))
        for name, value in (
            ("stall_timeout_ms", self.stall_timeout_ms),
            ("budget_poll_ms", self.budget_poll_ms),
            ("maximum_region_ms", self.maximum_region_ms),
            ("close_timeout_ms", self.close_timeout_ms),
        ):
            if isinstance(value, bool) or int(value) <= 0 or int(value) > 0xFFFFFFFF:
                raise ValueError("{} must be a positive uint32".format(name))
        if str(self.sdpa_backend).lower() not in {"math", "flash_attention"}:
            raise ValueError("sdpa_backend must be 'math' or 'flash_attention'")
        return self

    @property
    def chunk_bytes(self) -> int:
        return _parse_bytes(self.chunk_size, allow_auto=False)

    @property
    def cache_target_bytes(self) -> int:
        return _parse_bytes(self.cache_target, allow_auto=True)

    @property
    def headroom_bytes(self) -> int:
        return _parse_bytes(self.device_headroom, allow_auto=False)

    @property
    def scratch_bytes(self) -> int:
        return _parse_bytes(self.scratch_cap, allow_auto=False)

    @property
    def gemm_workspace_bytes(self) -> int:
        return _parse_bytes(self.gemm_workspace, allow_auto=False)


@dataclass(frozen=True)
class NativeErrorInfo:
    status: int
    native_code: int
    stage: str
    operation: str
    message: str


@dataclass(frozen=True)
class NativeLeaseInfo:
    lease: int
    state: int
    result: int
    live_tensor_storages: int
    resolved_ranges: int
    elapsed_milliseconds: float
    stream: int


@dataclass(frozen=True)
class NativeLeaseToken:
    """Exact generation token returned to the scheduler."""

    lease: int
    sequence: int


@dataclass(frozen=True)
class NativeGemmToken:
    """Already-retired synchronous tiled-GEMM generation."""

    sequence: int
    result: Mapping[str, Union[int, float]]


class _SessionConfigV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("policy", ctypes.c_uint32),
        ("chunk_bytes", ctypes.c_uint64),
        ("cache_target_bytes", ctypes.c_uint64),
        ("device_headroom_bytes", ctypes.c_uint64),
        ("scratch_arena_bytes", ctypes.c_uint64),
        ("staging_slots", ctypes.c_uint32),
        ("stall_timeout_milliseconds", ctypes.c_uint32),
        ("budget_poll_milliseconds", ctypes.c_uint32),
        ("maximum_transaction_milliseconds", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class _AllocationDescV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("hint", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _AllocationInfoV1(ctypes.Structure):
    _fields_ = _AllocationDescV1._fields_


class _AccessRangeV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("allocation", ctypes.c_uint64),
        ("byte_offset", ctypes.c_uint64),
        ("byte_length", ctypes.c_uint64),
        ("mode", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 2),
    ]


class _LeaseDescV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("ranges", ctypes.POINTER(_AccessRangeV1)),
        ("range_count", ctypes.c_size_t),
        ("reserved", ctypes.c_uint64 * 6),
    ]


class _LeaseInfoV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("result", ctypes.c_uint32),
        ("live_tensor_storages", ctypes.c_uint64),
        ("resolved_ranges", ctypes.c_uint64),
        ("elapsed_milliseconds", ctypes.c_double),
        ("stream", ctypes.c_size_t),
        ("reserved", ctypes.c_uint64 * 4),
    ]


_TELEMETRY_COUNTERS = (
    "allocations_created",
    "allocations_released",
    "handles_created",
    "handles_reused",
    "maps",
    "set_access_calls",
    "unmaps",
    "event_boundaries",
    "unsafe_remaps",
    "unsafe_transitions",
    "h2d_bytes",
    "d2h_bytes",
    "cache_hits",
    "cache_misses",
    "clean_evictions",
    "dirty_evictions",
    "dirty_writebacks",
    "prefetches",
    "leases_acquired",
    "leases_sealed",
    "leases_retired",
    "events_recorded",
    "events_retired",
    "watchdog_rejections",
    "budget_shrinks",
    "budget_grows",
    "resident_bytes",
    "resident_peak_bytes",
    "cache_target_bytes",
    "tensor_views_created",
    "tensor_views_live",
    "tensor_views_peak",
    "scratch_arena_bytes",
    "scratch_allocation_calls",
    "scratch_free_calls",
    "scratch_failures",
    "scratch_bytes_current",
    "scratch_bytes_peak",
)


class _TelemetryV1(ctypes.Structure):
    _fields_ = (
        [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32)]
        + [(name, ctypes.c_uint64) for name in _TELEMETRY_COUNTERS]
        + [
            ("stable_addresses", ctypes.c_uint32),
            ("no_physical_aliases", ctypes.c_uint32),
            ("quarantined", ctypes.c_uint32),
            ("reserved0", ctypes.c_uint32),
            ("reserved", ctypes.c_uint64 * 8),
        ]
    )


class _ErrorV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("native_code", ctypes.c_int64),
        ("stage", ctypes.c_char * 32),
        ("operation", ctypes.c_char * 64),
        ("message", ctypes.c_char * 256),
    ]


class _GemmMatrixV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("allocation", ctypes.c_uint64),
        ("byte_offset", ctypes.c_uint64),
        ("rows", ctypes.c_uint64),
        ("columns", ctypes.c_uint64),
        ("leading_dimension", ctypes.c_uint64),
        ("layout", ctypes.c_uint32),
        ("operation", ctypes.c_uint32),
        ("dtype", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _GemmProblemV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("a", _GemmMatrixV1),
        ("b", _GemmMatrixV1),
        ("c", _GemmMatrixV1),
        ("m", ctypes.c_uint64),
        ("n", ctypes.c_uint64),
        ("k", ctypes.c_uint64),
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("compute", ctypes.c_uint32),
        ("prefetch_distance", ctypes.c_uint32),
        ("workspace_bytes", ctypes.c_uint64),
        ("preferred_tile_m", ctypes.c_uint64),
        ("preferred_tile_n", ctypes.c_uint64),
        ("preferred_tile_k", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 6),
    ]


_GEMM_RESULT_COUNTERS = (
    "tile_count",
    "tiles_submitted",
    "tiles_completed",
    "events_recorded",
    "events_retired",
    "maps",
    "set_access_calls",
    "h2d_bytes",
    "d2h_bytes",
    "clean_evictions",
    "dirty_evictions",
    "handles_reused",
    "prefetches",
    "cublas_core_tiles",
    "cublas_lt_tiles",
    "algorithm_cache_hits",
    "workspace_bytes",
    "maximum_working_set_bytes",
    "tile_m",
    "tile_n",
    "tile_k",
)


class _GemmResultV1(ctypes.Structure):
    _fields_ = (
        [
            ("struct_size", ctypes.c_uint32),
            ("abi_version", ctypes.c_uint32),
            ("status", ctypes.c_uint32),
            ("boundary", ctypes.c_uint32),
            ("native_code", ctypes.c_int64),
        ]
        + [(name, ctypes.c_uint64) for name in _GEMM_RESULT_COUNTERS]
        + [
            ("maximum_kernel_milliseconds", ctypes.c_double),
            ("elapsed_milliseconds", ctypes.c_double),
            ("reserved", ctypes.c_uint64 * 8),
        ]
    )


_CALL = ctypes.CFUNCTYPE
_StatusNameFn = _CALL(ctypes.c_char_p, ctypes.c_uint32)
_SessionCreateFn = _CALL(ctypes.c_uint32, ctypes.POINTER(_SessionConfigV1), ctypes.POINTER(ctypes.c_uint64))
_SessionErrorFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_ErrorV1))
_SessionTelemetryFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_TelemetryV1))
_SessionCloseFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64)
_AllocationCreateFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_AllocationDescV1), ctypes.POINTER(ctypes.c_uint64))
_AllocationInfoFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_AllocationInfoV1))
_AllocationWriteFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
_AllocationReadFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)
_AllocationReleaseFn = _CALL(ctypes.c_uint32, ctypes.c_uint64)
_AllocationDiscardFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
_PrefetchFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_AccessRangeV1), ctypes.c_size_t)
_LeaseAcquireFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_LeaseDescV1), ctypes.POINTER(ctypes.c_uint64))
_LeaseInfoFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(_LeaseInfoV1))
_LeaseSealFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32)
_LeaseWaitFn = _CALL(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(_LeaseInfoV1))
_GemmExecuteFn = _CALL(
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.POINTER(_GemmProblemV1),
    ctypes.POINTER(_GemmResultV1),
)


class _ApiV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("status_name", _StatusNameFn),
        ("session_create", _SessionCreateFn),
        ("session_get_error", _SessionErrorFn),
        ("session_get_telemetry", _SessionTelemetryFn),
        ("session_close", _SessionCloseFn),
        ("allocation_create", _AllocationCreateFn),
        ("allocation_get_info", _AllocationInfoFn),
        ("allocation_write", _AllocationWriteFn),
        ("allocation_read", _AllocationReadFn),
        ("allocation_release", _AllocationReleaseFn),
        ("allocation_discard", _AllocationDiscardFn),
        ("session_prefetch", _PrefetchFn),
        ("lease_acquire", _LeaseAcquireFn),
        ("lease_get_info", _LeaseInfoFn),
        ("lease_seal", _LeaseSealFn),
        ("lease_poll", _LeaseInfoFn),
        ("lease_wait", _LeaseWaitFn),
        ("gemm_execute", _GemmExecuteFn),
        ("reserved", ctypes.c_uint64 * 15),
    ]


class _CtypesControl:
    """Typed, exception-based facade over ``xvram_torch_runtime_api_v1``."""

    def __init__(self, api: _ApiV1) -> None:
        if int(api.abi_version) != TORCH_RUNTIME_ABI_VERSION:
            raise NativeTorchUnavailableError(
                "load", 2, "native API version {} is unsupported".format(api.abi_version)
            )
        if int(api.struct_size) < _ApiV1.reserved.offset:
            raise NativeTorchUnavailableError(
                "load", 2, "native API prefix is too small"
            )
        self._api = api

    def status_name(self, status: int) -> str:
        try:
            value = self._api.status_name(int(status))
        except (ValueError, OSError):
            value = None
        return _decode(value) if value else "status_{}".format(status)

    def create_session(self, config: NativeRuntimeConfig) -> int:
        config.validate()
        native = _SessionConfigV1()
        _tag(native)
        native.device_ordinal = int(config.device)
        native.policy = _POLICIES[str(config.policy).lower()]
        native.chunk_bytes = config.chunk_bytes
        native.cache_target_bytes = config.cache_target_bytes
        native.device_headroom_bytes = config.headroom_bytes
        native.scratch_arena_bytes = config.scratch_bytes
        native.staging_slots = int(config.staging_slots)
        native.stall_timeout_milliseconds = int(config.stall_timeout_ms)
        native.budget_poll_milliseconds = int(config.budget_poll_ms)
        native.maximum_transaction_milliseconds = int(config.maximum_region_ms)
        output = ctypes.c_uint64(0)
        status = int(self._api.session_create(ctypes.byref(native), ctypes.byref(output)))
        self._check(status, "session_create")
        if output.value == 0:
            raise NativeTorchRuntimeError("session_create", 15, "native returned a null session")
        return int(output.value)

    def error(self, session: int) -> NativeErrorInfo:
        value = _ErrorV1()
        _tag(value)
        status = int(self._api.session_get_error(int(session), ctypes.byref(value)))
        if status != _STATUS_SUCCESS:
            return NativeErrorInfo(status, 0, "", "session_get_error", self.status_name(status))
        return NativeErrorInfo(
            int(value.status),
            int(value.native_code),
            _decode(value.stage),
            _decode(value.operation),
            _decode(value.message),
        )

    def telemetry(self, session: int) -> Dict[str, Union[int, bool]]:
        value = _TelemetryV1()
        _tag(value)
        status = int(self._api.session_get_telemetry(int(session), ctypes.byref(value)))
        self._check(status, "session_get_telemetry", session)
        if int(value.abi_version) != TORCH_RUNTIME_ABI_VERSION:
            raise NativeTorchRuntimeError("session_get_telemetry", 2, "telemetry ABI changed")
        result: Dict[str, Union[int, bool]] = {
            name: int(getattr(value, name)) for name in _TELEMETRY_COUNTERS
        }
        result.update(
            stable_addresses=bool(value.stable_addresses),
            no_physical_aliases=bool(value.no_physical_aliases),
            quarantined=bool(value.quarantined),
        )
        return result

    def close_session(self, session: int, timeout_ms: int) -> None:
        status = int(self._api.session_close(int(session), int(timeout_ms)))
        self._check(status, "session_close", session)

    def create_allocation(self, session: int, bytes_value: int, hint: str) -> int:
        value = _AllocationDescV1()
        _tag(value)
        value.bytes = _u64(bytes_value, "allocation bytes")
        value.hint = _HINTS[hint]
        output = ctypes.c_uint64(0)
        status = int(
            self._api.allocation_create(int(session), ctypes.byref(value), ctypes.byref(output))
        )
        self._check(status, "allocation_create", session)
        if output.value == 0:
            raise NativeTorchRuntimeError("allocation_create", 15, "native returned a null allocation")
        return int(output.value)

    def allocation_info(self, allocation: int) -> Tuple[int, int]:
        value = _AllocationInfoV1()
        _tag(value)
        status = int(self._api.allocation_get_info(int(allocation), ctypes.byref(value)))
        self._check(status, "allocation_get_info")
        return int(value.bytes), int(value.hint)

    def write(self, allocation: int, offset: int, pointer: int, length: int) -> None:
        status = int(
            self._api.allocation_write(
                int(allocation), _u64(offset, "write offset"), ctypes.c_void_p(pointer), _u64(length, "write length")
            )
        )
        self._check(status, "allocation_write")

    def read(self, allocation: int, offset: int, pointer: int, length: int) -> None:
        status = int(
            self._api.allocation_read(
                int(allocation), _u64(offset, "read offset"), ctypes.c_void_p(pointer), _u64(length, "read length")
            )
        )
        self._check(status, "allocation_read")

    def release(self, allocation: int) -> None:
        self._check(int(self._api.allocation_release(int(allocation))), "allocation_release")

    def discard(self, allocation: int, offset: int, length: int) -> None:
        self._check(
            int(self._api.allocation_discard(int(allocation), _u64(offset, "discard offset"), _u64(length, "discard length"))),
            "allocation_discard",
        )

    def prefetch(self, session: int, ranges: Sequence[Tuple[int, int, int, int]]) -> None:
        native, count = _native_ranges(ranges)
        status = int(self._api.session_prefetch(int(session), native, count))
        self._check(status, "session_prefetch", session)

    def acquire(self, session: int, ranges: Sequence[Tuple[int, int, int, int]]) -> NativeLeaseInfo:
        native, count = _native_ranges(ranges)
        description = _LeaseDescV1()
        _tag(description)
        description.ranges = native
        description.range_count = count
        output = ctypes.c_uint64(0)
        status = int(
            self._api.lease_acquire(int(session), ctypes.byref(description), ctypes.byref(output))
        )
        self._check(status, "lease_acquire", session)
        if output.value == 0:
            raise NativeTorchRuntimeError("lease_acquire", 15, "native returned a null lease")
        return self.lease_info(int(output.value))

    def lease_info(self, lease: int) -> NativeLeaseInfo:
        return self._observe(self._api.lease_get_info, lease, "lease_get_info")

    def seal(self, lease: int, mode: int) -> None:
        self._check(int(self._api.lease_seal(int(lease), int(mode))), "lease_seal")

    def poll(self, lease: int) -> NativeLeaseInfo:
        return self._observe(self._api.lease_poll, lease, "lease_poll")

    def wait(self, lease: int, timeout_ms: int) -> NativeLeaseInfo:
        value = _LeaseInfoV1()
        _tag(value)
        status = int(self._api.lease_wait(int(lease), int(timeout_ms), ctypes.byref(value)))
        self._check(status, "lease_wait")
        return _lease_snapshot(lease, value)

    def execute_gemm(
        self,
        session: int,
        problem: _GemmProblemV1,
    ) -> Dict[str, Union[int, float]]:
        result = _GemmResultV1()
        _tag(result)
        status = int(
            self._api.gemm_execute(int(session), ctypes.byref(problem), ctypes.byref(result))
        )
        if status != _STATUS_SUCCESS:
            self._check(status, "gemm_execute", session)
        if int(result.abi_version) != TORCH_RUNTIME_ABI_VERSION:
            raise NativeTorchRuntimeError("gemm_execute", 2, "GEMM result ABI changed")
        snapshot: Dict[str, Union[int, float]] = {
            "status": int(result.status),
            "boundary": int(result.boundary),
            "native_code": int(result.native_code),
            "maximum_kernel_milliseconds": float(result.maximum_kernel_milliseconds),
            "elapsed_milliseconds": float(result.elapsed_milliseconds),
        }
        snapshot.update({name: int(getattr(result, name)) for name in _GEMM_RESULT_COUNTERS})
        if int(result.status) != _GEMM_COMPLETED:
            raise NativeTorchRuntimeError(
                "gemm_execute",
                15,
                "native returned incomplete GEMM status {} at boundary {}".format(
                    result.status, result.boundary
                ),
                native_code=int(result.native_code),
            )
        return snapshot

    def _observe(self, function: Any, lease: int, operation: str) -> NativeLeaseInfo:
        value = _LeaseInfoV1()
        _tag(value)
        status = int(function(int(lease), ctypes.byref(value)))
        self._check(status, operation)
        return _lease_snapshot(lease, value)

    def _check(self, status: int, operation: str, session: int = 0) -> None:
        if int(status) == _STATUS_SUCCESS:
            return
        detail = self.status_name(status)
        native_code = 0
        if session:
            info = self.error(session)
            native_code = info.native_code
            pieces = [info.stage, info.operation, info.message]
            native_detail = ": ".join(item for item in pieces if item)
            if native_detail:
                detail = native_detail
        raise NativeTorchRuntimeError(operation, status, detail, native_code=native_code)


class _LoadedLibrary:
    def __init__(self, path: Path, torch_module: Any) -> None:
        self.path = path
        self.torch = torch_module
        self.dll_directories = _open_windows_dll_directories(path, torch_module)
        try:
            loader = getattr(getattr(torch_module, "ops", None), "load_library", None)
            if loader is None or not callable(loader):
                raise NativeTorchUnavailableError(
                    "load", 7, "PyTorch does not expose torch.ops.load_library"
                )
            loader(str(path))
            self.native = ctypes.CDLL(str(path))
            get_api = self.native.xvram_torch_runtime_get_api
            get_api.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
            get_api.restype = ctypes.c_uint32
            api = _ApiV1()
            status = int(
                get_api(TORCH_RUNTIME_ABI_VERSION, ctypes.sizeof(_ApiV1), ctypes.byref(api))
            )
            if status != _STATUS_SUCCESS:
                raise NativeTorchUnavailableError(
                    "get_api", status, "private v1 ABI is unavailable"
                )
            self.control = _CtypesControl(api)
        except BaseException:
            _close_dll_directories(self.dll_directories)
            raise

    def create_scratch_allocator(self) -> Any:
        memory = getattr(getattr(self.torch, "cuda", None), "memory", None)
        allocator_type = getattr(memory, "CUDAPluggableAllocator", None)
        if allocator_type is None or not callable(allocator_type):
            raise NativeTorchUnavailableError(
                "scratch_allocator", 7, "PyTorch does not expose CUDAPluggableAllocator"
            )
        return allocator_type(
            str(self.path),
            "xvram_torch_runtime_scratch_alloc",
            "xvram_torch_runtime_scratch_free",
        )


class NativeInferenceBackend(InferenceBackend):
    """Lease-safe native implementation of :class:`InferenceBackend`.

    The optional underscored arguments are dependency-injection seams for the
    no-GPU test suite; applications should use :class:`NativeBackendFactory`.
    """

    def __init__(
        self,
        *,
        config: Optional[NativeRuntimeConfig] = None,
        library_path: Optional[Union[str, os.PathLike[str]]] = None,
        _torch_module: Any = None,
        _control: Any = None,
        _library_owner: Any = None,
        _scratch_allocator: Any = None,
    ) -> None:
        self.config = (config or NativeRuntimeConfig()).validate()
        self._library_path = library_path
        self._torch = _torch_module
        self._control = _control
        self._library_owner = _library_owner
        self._scratch_allocator = _scratch_allocator
        self._plan: Optional[InferencePlan] = None
        self._session = 0
        self._owner_thread = 0
        self._allocations: Dict[str, int] = {}
        self._storage_bytes: Dict[str, int] = {}
        self._storage_kinds: Dict[str, ValueKind] = {}
        self._bound_inputs: set[str] = set()
        self._discarded_storages: set[str] = set()
        self._active_token: Optional[Union[NativeLeaseToken, NativeGemmToken]] = None
        self._gemm_results: List[Mapping[str, Union[int, float]]] = []
        self._closed = False
        self._prepared = False
        self._local_metrics: Dict[str, int] = {
            "nodes_submitted": 0,
            "nodes_retired": 0,
            "prefetch_calls": 0,
            "discard_calls": 0,
            "outputs_materialized": 0,
        }

    @property
    def session_handle(self) -> int:
        """Opaque session identity, suitable only for the private custom op."""

        return self._session

    def prepare(self, plan: InferencePlan, state_provider: StateProvider) -> None:
        if self._prepared:
            if self._plan is not plan and self._plan.graph_hash != plan.graph_hash:
                raise BackendContractError("native backend cannot be rebound to another graph")
            return
        if self._closed:
            raise BackendContractError("native backend is closed")
        self._owner_thread = threading.get_ident()
        self._ensure_torch_and_control()
        _initialize_primary_context(self._torch, int(self.config.device))
        self._reject_capture()
        self._session = int(self._control.create_session(self.config))
        self._plan = plan
        try:
            self._allocate_plan(plan)
            self._write_state(plan, state_provider)
            if any(node.adapter in _SCRATCH_ADAPTERS for node in plan.nodes):
                self._ensure_scratch_allocator()
        except BaseException:
            self._cleanup_after_prepare_failure()
            raise
        self._prepared = True

    def bind_input(self, value: ManagedValue, host_tensor: Any) -> None:
        self._require_ready()
        if value.kind is not ValueKind.INPUT:
            raise BackendContractError("bind_input accepts only planned user inputs")
        pointer, length = _contiguous_tensor_region(host_tensor, value)
        if length != value.spec.span_bytes:
            raise BackendContractError("input byte span differs from the static plan")
        allocation = self._allocation(value.storage_id)
        self._control.write(allocation, value.storage_offset_bytes, pointer, length)
        self._bound_inputs.add(value.name)
        self._discarded_storages.discard(value.storage_id)

    def prefetch(self, ranges: Tuple[ManagedRange, ...]) -> None:
        self._require_ready()
        self._require_no_active_lease()
        native = self._convert_ranges(ranges, widen_activation_writes=False)
        if native:
            self._control.prefetch(self._session, native)
            self._local_metrics["prefetch_calls"] += 1

    def submit(self, request: ExecutionRequest) -> NativeLeaseToken:
        self._require_ready()
        self._require_no_active_lease()
        self._reject_capture()
        node = request.node
        if node.adapter in {"linear_out", "mm_out"} and self._requires_tiled_gemm(request):
            return self._submit_tiled_gemm(request)
        native_ranges = self._convert_ranges(request.ranges, widen_activation_writes=True)
        lease_info = self._control.acquire(self._session, native_ranges)
        if lease_info.state != _LEASE_ARMED or lease_info.stream == 0:
            self._cancel_unsubmitted_lease(lease_info.lease)
            raise BackendContractError("native lease is not armed on a valid runtime stream")
        possible_submission = False
        try:
            stream = self._external_stream(lease_info.stream)
            with ExitStack() as contexts:
                contexts.enter_context(_inference_context(self._torch))
                contexts.enter_context(self._torch.cuda.stream(stream))
                self._assert_runtime_stream(lease_info.stream)
                views = self._create_node_views(lease_info.lease, node)
                try:
                    possible_submission = True
                    self._dispatch_node(node, views)
                finally:
                    views.clear()
            del stream
            self._assert_no_live_views(lease_info.lease)
            self._control.seal(lease_info.lease, _SEAL_SUCCESS)
        except BaseException:
            self._retire_failed_submission(lease_info.lease, possible_submission)
            raise
        token = NativeLeaseToken(lease=lease_info.lease, sequence=int(request.sequence))
        self._active_token = token
        self._discarded_storages.discard(node.output_value and self._plan.value(node.output_value).storage_id)
        self._local_metrics["nodes_submitted"] += 1
        return token

    def retire(self, token: Any) -> None:
        self._require_ready()
        if token != self._active_token:
            raise BackendContractError("retire token is stale, foreign, or already retired")
        if isinstance(token, NativeGemmToken):
            self._active_token = None
            self._local_metrics["nodes_retired"] += 1
            return
        if not isinstance(token, NativeLeaseToken):
            raise BackendContractError("retire token has an invalid native generation type")
        info = self._control.wait(token.lease, int(self.config.stall_timeout_ms))
        if info.state != _LEASE_COMPLETED or info.result != _STATUS_SUCCESS:
            raise NativeTorchRuntimeError(
                "lease_wait",
                info.result or _STATUS_TIMEOUT,
                "lease retired in state {}".format(info.state),
            )
        self._active_token = None
        self._local_metrics["nodes_retired"] += 1

    def release_values(self, names: Tuple[str, ...]) -> None:
        self._require_ready()
        self._require_no_active_lease()
        if self._plan is None:
            raise BackendContractError("backend has no plan")
        storages: set[str] = set()
        for name in names:
            value = self._plan.value(name)
            if value.kind is ValueKind.ACTIVATION and not value.is_alias:
                storages.add(value.storage_id)
        for storage_id in sorted(storages):
            if storage_id in self._discarded_storages:
                continue
            length = self._storage_bytes[storage_id]
            # An explicit non-zero length preserves the logical allocation; the
            # native (offset=0,length=0) form releases it permanently.
            self._control.discard(self._allocation(storage_id), 0, length)
            self._discarded_storages.add(storage_id)
            self._local_metrics["discard_calls"] += 1

    def read_output(self, value: ManagedValue) -> Any:
        self._require_ready()
        self._require_no_active_lease()
        if value.storage_id in self._discarded_storages:
            raise BackendContractError("cannot materialize a liveness-discarded output")
        dtype = _torch_dtype(self._torch, value.spec.dtype)
        output = self._torch.empty_strided(
            tuple(value.spec.sizes),
            tuple(value.spec.strides),
            dtype=dtype,
            device="cpu",
        )
        pointer = int(output.data_ptr())
        if pointer == 0:
            raise BackendContractError("PyTorch returned a null CPU output pointer")
        self._control.read(
            self._allocation(value.storage_id),
            value.storage_offset_bytes,
            pointer,
            value.spec.span_bytes,
        )
        self._local_metrics["outputs_materialized"] += 1
        return output

    def telemetry_snapshot(self) -> Dict[str, Union[int, bool]]:
        self._require_owner_thread()
        native: Dict[str, Union[int, bool]] = {}
        if self._session:
            native.update(self._control.telemetry(self._session))
        native.update(self._local_metrics)
        if self._gemm_results:
            native["gemm_calls"] = len(self._gemm_results)
            for name in _GEMM_RESULT_COUNTERS:
                native["gemm_" + name] = sum(int(item[name]) for item in self._gemm_results)
            native["gemm_maximum_kernel_milliseconds"] = max(
                float(item["maximum_kernel_milliseconds"]) for item in self._gemm_results
            )
            native["gemm_elapsed_milliseconds"] = sum(
                float(item["elapsed_milliseconds"]) for item in self._gemm_results
            )
        return native

    def close(self) -> None:
        if self._closed:
            return
        self._require_owner_thread(allow_unowned=True)
        errors: List[BaseException] = []
        if self._active_token is not None:
            try:
                self.retire(self._active_token)
            except BaseException as error:
                errors.append(error)
        for storage_id, allocation in reversed(tuple(self._allocations.items())):
            try:
                self._control.release(allocation)
            except BaseException as error:
                errors.append(error)
            finally:
                self._allocations.pop(storage_id, None)
        if self._session:
            try:
                self._control.close_session(self._session, int(self.config.close_timeout_ms))
            except BaseException as error:
                errors.append(error)
            self._session = 0
        self._closed = True
        self._prepared = False
        self._scratch_allocator = None
        if errors:
            raise errors[0]

    def _ensure_torch_and_control(self) -> None:
        if self._torch is None:
            self._torch = _import_torch()
        if self._control is not None:
            return
        path = find_runtime_library(self._library_path)
        owner = _LoadedLibrary(path, self._torch)
        self._library_owner = owner
        self._control = owner.control

    def _ensure_scratch_allocator(self) -> None:
        if self._scratch_allocator is not None:
            return
        if self._library_owner is None:
            raise NativeTorchUnavailableError(
                "scratch_allocator",
                7,
                "scratch adapters require the loaded native library owner",
            )
        clear_cublas = getattr(
            getattr(self._torch, "_C", None), "_cuda_clearCublasWorkspaces", None
        )
        if clear_cublas is None or not callable(clear_cublas):
            raise NativeTorchUnavailableError(
                "scratch_allocator",
                7,
                "PyTorch cannot retire lease-scoped cuBLAS workspaces",
            )
        self._scratch_allocator = self._library_owner.create_scratch_allocator()

    def _allocate_plan(self, plan: InferencePlan) -> None:
        for value in sorted(plan.values.values(), key=lambda item: (item.storage_id, item.name)):
            prior = self._storage_bytes.get(value.storage_id)
            if prior is not None:
                if prior != value.storage_bytes:
                    raise BackendContractError("one logical storage has conflicting capacities")
                continue
            hint = "hot" if value.is_persistent else (
                "streaming" if value.kind is ValueKind.ACTIVATION else "normal"
            )
            handle = self._control.create_allocation(
                self._session, int(value.storage_bytes), hint
            )
            self._allocations[value.storage_id] = int(handle)
            self._storage_bytes[value.storage_id] = int(value.storage_bytes)
            self._storage_kinds[value.storage_id] = value.kind

    def _write_state(self, plan: InferencePlan, provider: StateProvider) -> None:
        groups: Dict[str, List[Tuple[str, ManagedValue, Any]]] = {}
        for name in sorted(plan.state_targets):
            value = plan.value(name)
            target = plan.state_targets[name]
            info = provider.describe(target)
            if int(info.storage_bytes) != int(value.storage_bytes):
                raise BackendContractError("state storage capacity changed after capture")
            groups.setdefault(value.storage_id, []).append((target, value, info))

        for storage_id in sorted(groups):
            entries = groups[storage_id]
            first_info = entries[0][2]
            storage_method = getattr(first_info.tensor, "untyped_storage", None)
            if storage_method is not None and callable(storage_method):
                # Ordinary CPU modules take the one-call full-storage fast path.
                storage = storage_method()
                pointer = int(storage.data_ptr())
                if pointer == 0:
                    raise BackendContractError("state provider returned a null storage pointer")
                self._control.write(
                    self._allocation(storage_id), 0, pointer, int(first_info.storage_bytes)
                )
                continue
            self._write_chunked_state_storage(
                storage_id,
                entries,
                provider,
                int(first_info.storage_bytes),
            )

    def _write_chunked_state_storage(
        self,
        storage_id: str,
        entries: Sequence[Tuple[str, ManagedValue, Any]],
        provider: StateProvider,
        storage_bytes: int,
    ) -> None:
        covered: List[Tuple[int, int]] = []
        allocation = self._allocation(storage_id)
        for target, _value, info in entries:
            logical_bytes = int(info.spec.numel) * int(info.spec.element_size)
            target_begin = int(info.offset_bytes)
            target_end = target_begin + logical_bytes
            if (
                logical_bytes <= 0
                or target_begin < 0
                or target_end < target_begin
                or target_end > storage_bytes
            ):
                raise BackendContractError("chunked state target exceeds its shared storage")
            for chunk_offset, data in _iter_state_chunks(
                provider,
                target,
                info.tensor,
                logical_bytes,
                self.config.chunk_bytes,
            ):
                pointer, length = _byte_tensor_region(data)
                chunk_begin = target_begin + int(chunk_offset)
                chunk_end = chunk_begin + length
                if (
                    chunk_offset < 0
                    or length <= 0
                    or chunk_end < chunk_begin
                    or chunk_begin < target_begin
                    or chunk_end > target_end
                ):
                    raise BackendContractError("chunked state provider returned an invalid range")
                for begin, end in _subtract_covered(chunk_begin, chunk_end, covered):
                    self._control.write(
                        allocation,
                        begin,
                        pointer + (begin - chunk_begin),
                        end - begin,
                    )
                    covered = _add_covered(covered, begin, end)
        if covered != [(0, storage_bytes)]:
            raise BackendContractError(
                "chunked/tied state targets do not completely initialize logical storage"
            )

    def _convert_ranges(
        self,
        ranges: Iterable[ManagedRange],
        *,
        widen_activation_writes: bool,
    ) -> Tuple[Tuple[int, int, int, int], ...]:
        output: List[Tuple[int, int, int, int]] = []
        for item in ranges:
            offset = int(item.offset_bytes)
            length = int(item.length_bytes)
            if (
                widen_activation_writes
                and item.mode is AccessMode.WRITE_ONLY
                and self._storage_kinds.get(item.storage_id) is ValueKind.ACTIVATION
            ):
                # Activation slots are liveness-discarded by chunk.  Widening a
                # fresh write to the complete logical slot proves that no dead
                # host byte has to be preserved by H2D on the next generation.
                offset = 0
                length = self._storage_bytes[item.storage_id]
            output.append(
                (self._allocation(item.storage_id), offset, length, _ACCESS_MODES[item.mode])
            )
        return tuple(output)

    def _create_node_views(self, lease: int, node: Any) -> Dict[str, Any]:
        if self._plan is None:
            raise BackendContractError("backend has no plan")
        names = set(_argument_value_names(node.args))
        names.update(_argument_value_names(node.kwargs))
        names.add(node.output_value)
        views: Dict[str, Any] = {}
        wrapper = _resolved_wrapper(self._torch)
        for name in sorted(names):
            value = self._plan.value(name)
            scalar_type = _scalar_type(value.spec.dtype)
            views[name] = wrapper(
                int(self._session),
                int(lease),
                int(self._allocation(value.storage_id)),
                int(value.storage_offset_bytes),
                list(value.spec.sizes),
                list(value.spec.strides),
                scalar_type,
            )
        return views

    def _dispatch_node(self, node: Any, views: Mapping[str, Any]) -> None:
        adapter = str(node.adapter)
        args = _resolve_argument(node.args, views, self._torch)
        kwargs = dict(_resolve_argument(node.kwargs, views, self._torch))
        output = views[node.output_value]
        if adapter in _BASIC_OUT_ADAPTERS:
            operation = _aten_overload(self._torch, _BASIC_OUT_ADAPTERS[adapter], "out")
            result = operation(*args, **kwargs, out=output)
            del result, args, kwargs, output
            return
        if adapter == "copy_out":
            operation = _aten_overload(self._torch, "clone", "out")
            result = operation(*args, **kwargs, out=output)
            del result, args, kwargs, output
            return
        if adapter == "cast_copy_out":
            operation = _aten_overload(self._torch, "_to_copy", "out")
            # The managed output dtype is authoritative; dtype/device kwargs
            # belong to the allocation-producing default overload.
            kwargs.pop("dtype", None)
            kwargs.pop("device", None)
            kwargs.pop("layout", None)
            kwargs.pop("pin_memory", None)
            result = operation(*args, **kwargs, out=output)
            del result, args, kwargs, output
            return
        if adapter == "reshape_copy":
            operation = _aten_overload(self._torch, "view_copy", "out")
            result = operation(*args, **kwargs, out=output)
            del result, args, kwargs, output
            return
        if adapter in _SCRATCH_ADAPTERS:
            self._dispatch_scratch(adapter, args, kwargs, output)
            del args, kwargs, output
            return
        raise BackendContractError(
            "no native adapter exists for planned operator {!r}".format(adapter)
        )

    def _dispatch_scratch(
        self, adapter: str, args: Any, kwargs: Dict[str, Any], output: Any
    ) -> None:
        self._ensure_scratch_allocator()
        cuda = self._torch.cuda
        pool_type = getattr(cuda, "MemPool", None)
        use_pool = getattr(cuda, "use_mem_pool", None)
        if pool_type is None or not callable(pool_type) or use_pool is None or not callable(use_pool):
            raise NativeTorchUnavailableError(
                "scratch_adapter", 7, "PyTorch does not expose MemPool/use_mem_pool"
            )
        allocator_handle = self._scratch_allocator.allocator()
        try:
            pool = pool_type(allocator=allocator_handle, no_split=True)
        except TypeError:
            pool = pool_type(allocator=allocator_handle)
        try:
            with use_pool(pool, device=int(self.config.device)):
                if adapter == "rms_norm_scratch_copy":
                    scratch = _aten_overload(self._torch, "rms_norm", "default")(
                        *args, **kwargs
                    )
                elif adapter == "sdpa_math_scratch_copy":
                    with self._sdpa_context():
                        scratch = _aten_overload(
                            self._torch, "scaled_dot_product_attention", "default"
                        )(*args, **kwargs)
                else:  # pragma: no cover - guarded by caller
                    raise BackendContractError("unknown scratch adapter")
                copied = _aten_overload(self._torch, "copy", "out")(
                    output, scratch, False, out=output
                )
                del copied, scratch
                current_stream = getattr(cuda, "current_stream", None)
                if current_stream is not None and callable(current_stream):
                    selected_stream = current_stream(device=int(self.config.device))
                    synchronize = getattr(selected_stream, "synchronize", None)
                    if synchronize is not None and callable(synchronize):
                        # The PyTorch caching-pool layer may defer a temporary
                        # block's plugin free callback until its stream work is
                        # complete.  A scratch adapter is deliberately a
                        # bounded region; drain it here so the native arena is
                        # empty before the runtime records the lease boundary.
                        synchronize()
                    del selected_stream
                clear_cublas = getattr(
                    getattr(self._torch, "_C", None),
                    "_cuda_clearCublasWorkspaces",
                    None,
                )
                # PyTorch retains one cuBLAS workspace per handle/stream;
                # under the lease-bounded allocator that cache entry would
                # otherwise point into an arena about to be unmapped.  The
                # callable is preflighted when scratch adapters are prepared.
                clear_cublas()
                empty_cache = getattr(cuda, "empty_cache", None)
                if empty_cache is not None and callable(empty_cache):
                    # SDPA may leave a block in the temporary MemPool cache
                    # even after its result is destroyed.  Emptying while this
                    # pool is selected returns every block through the native
                    # free callback before lease sealing.
                    empty_cache()
        finally:
            # A per-region pool prevents cached scratch blocks from surviving
            # into lease sealing.  CPython destroys it immediately; collect is
            # a conservative fallback for Python implementations with cycles.
            del pool
            gc.collect()

    def _sdpa_context(self) -> Any:
        attention = getattr(getattr(self._torch, "nn", None), "attention", None)
        factory = getattr(attention, "sdpa_kernel", None)
        backends = getattr(attention, "SDPBackend", None)
        if factory is None or not callable(factory) or backends is None:
            raise NativeTorchUnavailableError(
                "sdpa_kernel", 7, "PyTorch does not expose the explicit SDPA backend API"
            )
        name = "MATH" if str(self.config.sdpa_backend).lower() == "math" else "FLASH_ATTENTION"
        selected = getattr(backends, name, None)
        if selected is None:
            raise NativeTorchUnavailableError(
                "sdpa_kernel", 7, "requested SDPA backend {} is unavailable".format(name)
            )
        return factory(backends=[selected])

    def _external_stream(self, identity: int) -> Any:
        constructor = getattr(self._torch.cuda, "ExternalStream", None)
        if constructor is None or not callable(constructor):
            raise NativeTorchUnavailableError(
                "external_stream", 7, "PyTorch does not expose torch.cuda.ExternalStream"
            )
        return constructor(int(identity), device=int(self.config.device))

    def _assert_runtime_stream(self, identity: int) -> None:
        current_stream = getattr(self._torch.cuda, "current_stream", None)
        if current_stream is None or not callable(current_stream):
            return
        current = current_stream(device=int(self.config.device))
        observed = getattr(current, "cuda_stream", None)
        if observed is not None and int(observed) != int(identity):
            raise BackendContractError("ATen did not enter the runtime-owned CUDA stream")

    def _assert_no_live_views(self, lease: int) -> None:
        info = self._control.lease_info(lease)
        if info.live_tensor_storages:
            gc.collect()
            info = self._control.lease_info(lease)
        if info.live_tensor_storages:
            raise NativeTorchRuntimeError(
                "lease_seal",
                _STATUS_VIEWS_LIVE,
                "{} managed tensor views remain live".format(info.live_tensor_storages),
            )

    def _cancel_unsubmitted_lease(self, lease: int) -> None:
        try:
            self._control.seal(lease, _SEAL_CANCELLED)
            self._control.wait(lease, int(self.config.stall_timeout_ms))
        except BaseException:
            pass

    def _retire_failed_submission(self, lease: int, possible_submission: bool) -> None:
        mode = _SEAL_FAILED_AFTER_SUBMISSION if possible_submission else _SEAL_CANCELLED
        try:
            self._control.seal(lease, mode)
        except BaseException:
            return
        try:
            self._control.wait(lease, int(self.config.stall_timeout_ms))
        except BaseException:
            return

    def _requires_tiled_gemm(self, request: ExecutionRequest) -> bool:
        total = sum(int(item.length_bytes) for item in request.ranges)
        telemetry = self._control.telemetry(self._session)
        target = int(telemetry.get("cache_target_bytes", 0))
        return target > 0 and total > target

    def _submit_tiled_gemm(self, request: ExecutionRequest) -> NativeGemmToken:
        execute = getattr(self._control, "execute_gemm", None)
        if execute is None or not callable(execute):
            raise BackendContractError(
                "oversized {} requires native v1 gemm_execute; full-working-set lease is forbidden".format(
                    request.node.adapter
                )
            )
        problem = self._make_gemm_problem(request)
        result = execute(self._session, problem)
        required = {"tiles_submitted", "tiles_completed", "events_recorded", "events_retired"}
        if not isinstance(result, Mapping) or not required.issubset(result):
            raise BackendContractError("native gemm_execute returned an invalid result")
        if int(result["tiles_submitted"]) != int(result["tiles_completed"]):
            raise BackendContractError("native gemm_execute did not retire every submitted tile")
        if int(result["events_recorded"]) != int(result["events_retired"]):
            raise BackendContractError("native gemm_execute left an event generation unretired")
        self._gemm_results.append(dict(result))
        token = NativeGemmToken(sequence=int(request.sequence), result=dict(result))
        self._active_token = token
        if self._plan is not None:
            self._discarded_storages.discard(
                self._plan.value(request.node.output_value).storage_id
            )
        self._local_metrics["nodes_submitted"] += 1
        return token

    def _make_gemm_problem(self, request: ExecutionRequest) -> _GemmProblemV1:
        if self._plan is None:
            raise BackendContractError("backend has no plan")
        node = request.node
        if node.adapter not in {"linear_out", "mm_out"}:
            raise BackendContractError("tiled GEMM accepts only mm or bias-free linear")
        arguments = tuple(node.args)
        if len(arguments) < 2 or not all(
            isinstance(item, ValueReference) for item in arguments[:2]
        ):
            raise BackendContractError("GEMM tensor operands are malformed")
        if node.adapter == "linear_out" and len(arguments) > 2 and arguments[2] is not None:
            raise BackendContractError("tiled linear supports only bias-free graphs")
        left = self._plan.value(arguments[0].name)
        right = self._plan.value(arguments[1].name)
        output = self._plan.value(node.output_value)
        if node.adapter == "linear_out":
            if len(right.spec.sizes) != 2 or len(left.spec.sizes) < 2:
                raise BackendContractError("tiled linear matrix ranks are invalid")
            k = int(left.spec.sizes[-1])
            n = int(right.spec.sizes[0])
            if int(right.spec.sizes[1]) != k or int(output.spec.sizes[-1]) != n:
                raise BackendContractError("tiled linear dimensions are inconsistent")
            m = int(left.spec.numel) // k
            a = self._gemm_matrix(left, flatten=True, transpose=False)
            b = self._gemm_matrix(right, flatten=False, transpose=True)
            c = self._gemm_matrix(output, flatten=True, transpose=False)
        else:
            if len(left.spec.sizes) != 2 or len(right.spec.sizes) != 2:
                raise BackendContractError("tiled mm requires rank-two operands")
            m, k = (int(item) for item in left.spec.sizes)
            right_k, n = (int(item) for item in right.spec.sizes)
            if right_k != k or output.spec.sizes != (m, n):
                raise BackendContractError("tiled mm dimensions are inconsistent")
            a = self._gemm_matrix(left, flatten=False, transpose=False)
            b = self._gemm_matrix(right, flatten=False, transpose=False)
            c = self._gemm_matrix(output, flatten=False, transpose=False)

        problem = _GemmProblemV1()
        _tag(problem)
        problem.a = a
        problem.b = b
        problem.c = c
        problem.m = m
        problem.n = n
        problem.k = k
        problem.alpha = 1.0
        problem.beta = 0.0
        problem.compute = 3 if output.spec.dtype == "torch.float64" else 1
        problem.prefetch_distance = int(self.config.prefetch_distance)
        problem.workspace_bytes = int(self.config.gemm_workspace_bytes)
        problem.preferred_tile_m = int(self.config.gemm_tile_m)
        problem.preferred_tile_n = int(self.config.gemm_tile_n)
        problem.preferred_tile_k = int(self.config.gemm_tile_k)
        return problem

    def _gemm_matrix(
        self, value: ManagedValue, *, flatten: bool, transpose: bool
    ) -> _GemmMatrixV1:
        try:
            dtype = _GEMM_DTYPES[value.spec.dtype]
        except KeyError as error:
            raise BackendContractError("native tiled GEMM dtype is unsupported") from error
        sizes = tuple(int(item) for item in value.spec.sizes)
        strides = tuple(int(item) for item in value.spec.strides)
        if flatten:
            rows = int(value.spec.numel) // sizes[-1]
            columns = sizes[-1]
            if strides != _dense_strides(sizes):
                raise BackendContractError("flattened GEMM operands must be dense row-major")
            layout = 1
            leading = columns
        else:
            if len(sizes) != 2:
                raise BackendContractError("GEMM matrix must be rank two")
            rows, columns = sizes
            if strides == (columns, 1):
                layout = 1
                leading = int(strides[0])
            elif strides == (1, rows):
                layout = 2
                leading = int(strides[1])
            else:
                raise BackendContractError("GEMM matrix must be dense row- or column-major")
        matrix = _GemmMatrixV1()
        _tag(matrix)
        matrix.allocation = self._allocation(value.storage_id)
        matrix.byte_offset = int(value.storage_offset_bytes)
        matrix.rows = rows
        matrix.columns = columns
        matrix.leading_dimension = leading
        matrix.layout = layout
        matrix.operation = 2 if transpose else 1
        matrix.dtype = dtype
        return matrix

    def _reject_capture(self) -> None:
        checker = getattr(self._torch.cuda, "is_current_stream_capturing", None)
        if checker is not None and callable(checker) and bool(checker()):
            raise BackendContractError("CUDA Graph capture is outside Phase 4b scope")

    def _cleanup_after_prepare_failure(self) -> None:
        errors: List[BaseException] = []
        for allocation in reversed(tuple(self._allocations.values())):
            try:
                self._control.release(allocation)
            except BaseException as error:
                errors.append(error)
        self._allocations.clear()
        if self._session:
            try:
                self._control.close_session(self._session, int(self.config.close_timeout_ms))
            except BaseException as error:
                errors.append(error)
        self._session = 0
        if errors:
            raise errors[0]

    def _allocation(self, storage_id: str) -> int:
        try:
            return self._allocations[storage_id]
        except KeyError as error:
            raise BackendContractError(
                "logical storage {!r} was not allocated".format(storage_id)
            ) from error

    def _require_ready(self) -> None:
        self._require_owner_thread()
        if not self._prepared or self._closed or not self._session:
            raise BackendContractError("native backend is not prepared")

    def _require_no_active_lease(self) -> None:
        if self._active_token is not None:
            raise BackendContractError("one compute lease is already active")

    def _require_owner_thread(self, *, allow_unowned: bool = False) -> None:
        if self._owner_thread == 0 and allow_unowned:
            return
        if self._owner_thread and threading.get_ident() != self._owner_thread:
            raise BackendContractError("native session calls must stay on their owner thread")


class NativeBackendFactory(InferenceBackendFactory):
    """Create independent native backends for validated inference plans."""

    def __init__(
        self,
        *,
        config: Optional[NativeRuntimeConfig] = None,
        library_path: Optional[Union[str, os.PathLike[str]]] = None,
        _torch_module: Any = None,
        _control_factory: Any = None,
    ) -> None:
        self.config = (config or NativeRuntimeConfig()).validate()
        self.library_path = library_path
        self._torch_module = _torch_module
        self._control_factory = _control_factory

    def create(self, plan: InferencePlan) -> NativeInferenceBackend:
        del plan
        control = self._control_factory() if self._control_factory is not None else None
        return NativeInferenceBackend(
            config=self.config,
            library_path=self.library_path,
            _torch_module=self._torch_module,
            _control=control,
        )


def find_runtime_library(
    library_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Path:
    """Resolve the private runtime from an explicit path, environment, or package."""

    configured: Optional[Union[str, os.PathLike[str]]] = library_path
    if configured is None:
        configured = os.environ.get(XVRAM_TORCH_RUNTIME_ENV)
    if configured is not None:
        return _resolve_library_path(configured)

    package = Path(__file__).resolve().parent
    names = (
        "xvram_torch_runtime.dll",
        "libxvram_torch_runtime.so",
        "libxvram_torch_runtime.dylib",
    )
    directories = (package, package / ".libs", package.parent / "lib", package.parent / "bin")
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    discovered = ctypes.util.find_library("xvram_torch_runtime")
    if discovered:
        candidate = Path(discovered)
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
    raise NativeTorchUnavailableError(
        "find_library",
        3,
        "set {} to the absolute native runtime path".format(XVRAM_TORCH_RUNTIME_ENV),
    )


def _parse_bytes(value: Union[int, str], *, allow_auto: bool) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a byte size")
    if isinstance(value, int):
        if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
            raise ValueError("byte size exceeds uint64")
        return value
    text = str(value).strip()
    if allow_auto and text.lower() == "auto":
        return 0
    units = {
        "": 1,
        "b": 1,
        "kb": 1000,
        "kib": 1 << 10,
        "mb": 1000**2,
        "mib": 1 << 20,
        "gb": 1000**3,
        "gib": 1 << 30,
    }
    lower = text.lower()
    suffix = ""
    for candidate in sorted(units, key=len, reverse=True):
        if candidate and lower.endswith(candidate):
            suffix = candidate
            lower = lower[: -len(candidate)].strip()
            break
    try:
        number = int(lower, 0)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid byte size {!r}".format(value)) from error
    result = number * units[suffix]
    if number < 0 or result > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("byte size exceeds uint64")
    return result


def _resolve_library_path(path_value: Union[str, os.PathLike[str]]) -> Path:
    try:
        path = Path(os.fspath(path_value)).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, TypeError) as error:
        raise NativeTorchUnavailableError(
            "find_library", 3, "native runtime does not exist: {!r}".format(path_value)
        ) from error
    if not path.is_file() or not path.is_absolute():
        raise NativeTorchUnavailableError(
            "find_library", 3, "native runtime path is not an absolute file"
        )
    return path


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise NativeTorchUnavailableError(
            "import_torch", 7, "a CUDA-enabled PyTorch installation is required"
        ) from error


def _initialize_primary_context(torch_module: Any, device: int) -> None:
    cuda = getattr(torch_module, "cuda", None)
    available = getattr(cuda, "is_available", None)
    if cuda is None or available is None or not callable(available) or not bool(available()):
        raise NativeTorchUnavailableError("cuda_init", 7, "PyTorch CUDA is unavailable")
    set_device = getattr(cuda, "set_device", None)
    initialize = getattr(cuda, "init", None)
    if set_device is None or not callable(set_device):
        raise NativeTorchUnavailableError("cuda_init", 7, "torch.cuda.set_device is unavailable")
    set_device(int(device))
    if initialize is not None and callable(initialize):
        initialize()
    current = getattr(cuda, "current_device", None)
    if current is not None and callable(current) and int(current()) != int(device):
        raise NativeTorchUnavailableError("cuda_init", 7, "PyTorch selected a different CUDA device")
    # Force primary-context materialization before the native runtime attaches.
    empty = getattr(torch_module, "empty", None)
    if empty is None or not callable(empty):
        raise NativeTorchUnavailableError("cuda_init", 7, "torch.empty is unavailable")
    probe = empty(0, device="cuda:{}".format(device))
    del probe


def _inference_context(torch_module: Any) -> Any:
    factory = getattr(torch_module, "inference_mode", None)
    return factory() if factory is not None and callable(factory) else nullcontext()


def _native_ranges(
    ranges: Sequence[Tuple[int, int, int, int]],
) -> Tuple[Any, int]:
    if not ranges:
        raise ValueError("at least one native access range is required")
    array_type = _AccessRangeV1 * len(ranges)
    array = array_type()
    for index, (allocation, offset, length, mode) in enumerate(ranges):
        item = array[index]
        _tag(item)
        item.allocation = _u64(allocation, "allocation")
        item.byte_offset = _u64(offset, "range offset")
        item.byte_length = _u64(length, "range length")
        item.mode = int(mode)
        if item.allocation == 0 or item.byte_length == 0 or item.mode not in (1, 2, 3):
            raise ValueError("invalid native access range")
    return array, len(ranges)


def _lease_snapshot(lease: int, value: _LeaseInfoV1) -> NativeLeaseInfo:
    if int(value.abi_version) != TORCH_RUNTIME_ABI_VERSION:
        raise NativeTorchRuntimeError("lease_info", 2, "lease info ABI changed")
    return NativeLeaseInfo(
        lease=int(lease),
        state=int(value.state),
        result=int(value.result),
        live_tensor_storages=int(value.live_tensor_storages),
        resolved_ranges=int(value.resolved_ranges),
        elapsed_milliseconds=float(value.elapsed_milliseconds),
        stream=int(value.stream),
    )


def _tag(value: ctypes.Structure) -> None:
    value.struct_size = ctypes.sizeof(type(value))
    value.abi_version = TORCH_RUNTIME_ABI_VERSION


def _u64(value: int, name: str) -> int:
    result = int(value)
    if result < 0 or result > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("{} exceeds uint64".format(name))
    return result


def _decode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _open_windows_dll_directories(path: Path, torch_module: Any) -> Tuple[Any, ...]:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return ()
    candidates = [path.parent]
    torch_file = getattr(torch_module, "__file__", None)
    if torch_file:
        candidates.append(Path(torch_file).resolve().parent / "lib")
    handles: List[Any] = []
    seen: set[str] = set()
    try:
        for candidate in candidates:
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            if key in seen or not resolved.is_dir():
                continue
            handles.append(os.add_dll_directory(str(resolved)))
            seen.add(key)
    except BaseException:
        _close_dll_directories(tuple(handles))
        raise
    return tuple(handles)


def _close_dll_directories(handles: Tuple[Any, ...]) -> None:
    for handle in reversed(handles):
        close = getattr(handle, "close", None)
        if close is not None and callable(close):
            close()


def _contiguous_tensor_region(tensor: Any, value: ManagedValue) -> Tuple[int, int]:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type) != "cpu":
        raise BackendContractError("native backing accepts only pageable CPU tensors")
    sizes = tuple(int(item) for item in tensor.size())
    strides = tuple(int(item) for item in tensor.stride())
    if sizes != value.spec.sizes or strides != value.spec.strides:
        raise BackendContractError("host tensor geometry changed after static validation")
    if strides != _dense_strides(sizes):
        raise BackendContractError("native inputs must be contiguous row-major tensors")
    storage = tensor.untyped_storage()
    base = int(storage.data_ptr())
    offset_method = getattr(tensor, "storage_offset", None)
    offset_elements = int(offset_method() if callable(offset_method) else 0)
    pointer = base + offset_elements * int(value.spec.element_size)
    if pointer <= 0:
        raise BackendContractError("host tensor has a null storage pointer")
    return pointer, int(value.spec.span_bytes)


def _iter_state_chunks(
    provider: StateProvider,
    target: str,
    source: Any,
    logical_bytes: int,
    chunk_bytes: int,
) -> Iterable[Tuple[int, Any]]:
    iterator = getattr(source, "iter_chunks", None)
    if iterator is None or not callable(iterator):
        iterator = getattr(provider, "iter_chunks", None)
        if iterator is not None and callable(iterator):
            chunks = iterator(target, chunk_bytes)
        else:
            reader = getattr(source, "read_bytes", None)
            if reader is None or not callable(reader):
                provider_reader = getattr(provider, "read_bytes", None)
                if provider_reader is None or not callable(provider_reader):
                    raise BackendContractError(
                        "chunked state must expose iter_chunks() or read_bytes()"
                    )

                def reader(offset: int, length: int) -> Any:
                    return provider_reader(target, offset, length)

            def generated() -> Iterable[Tuple[int, Any]]:
                offset = 0
                while offset < logical_bytes:
                    length = min(chunk_bytes, logical_bytes - offset)
                    yield offset, reader(offset, length)
                    offset += length

            chunks = generated()
    else:
        chunks = iterator(chunk_bytes)

    for chunk in chunks:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            offset, data = chunk
        else:
            offset = getattr(chunk, "offset_bytes", None)
            data = getattr(chunk, "data", None)
        if offset is None or data is None:
            raise BackendContractError("chunked state yielded malformed chunk metadata")
        yield int(offset), data


def _byte_tensor_region(data: Any) -> Tuple[int, int]:
    device = getattr(data, "device", None)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type) != "cpu":
        raise BackendContractError("state chunks must be pageable CPU tensors")
    contiguous = getattr(data, "is_contiguous", None)
    if contiguous is not None and callable(contiguous) and not bool(contiguous()):
        raise BackendContractError("state chunks must be contiguous")
    dtype = str(getattr(data, "dtype", ""))
    element_size_method = getattr(data, "element_size", None)
    numel_method = getattr(data, "numel", None)
    data_ptr = getattr(data, "data_ptr", None)
    if (
        not callable(element_size_method)
        or not callable(numel_method)
        or not callable(data_ptr)
    ):
        raise BackendContractError("state chunk is not tensor-like")
    element_size = int(element_size_method())
    length = int(numel_method()) * element_size
    if dtype and dtype != "torch.uint8" and element_size != 1:
        raise BackendContractError("state chunks must expose a byte-addressable dtype")
    pointer = int(data_ptr())
    if pointer <= 0 or length <= 0:
        raise BackendContractError("state chunk has an invalid CPU byte range")
    return pointer, length


def _subtract_covered(
    begin: int, end: int, covered: Sequence[Tuple[int, int]]
) -> Tuple[Tuple[int, int], ...]:
    pending = [(int(begin), int(end))]
    for prior_begin, prior_end in covered:
        updated: List[Tuple[int, int]] = []
        for current_begin, current_end in pending:
            if prior_end <= current_begin or prior_begin >= current_end:
                updated.append((current_begin, current_end))
                continue
            if current_begin < prior_begin:
                updated.append((current_begin, prior_begin))
            if prior_end < current_end:
                updated.append((prior_end, current_end))
        pending = updated
    return tuple(pending)


def _add_covered(
    covered: Sequence[Tuple[int, int]], begin: int, end: int
) -> List[Tuple[int, int]]:
    ranges = sorted(tuple(covered) + ((int(begin), int(end)),))
    merged: List[Tuple[int, int]] = []
    for current_begin, current_end in ranges:
        if not merged or current_begin > merged[-1][1]:
            merged.append((current_begin, current_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], current_end))
    return merged


def _dense_strides(sizes: Sequence[int]) -> Tuple[int, ...]:
    stride = 1
    result: List[int] = []
    for size in reversed(tuple(int(item) for item in sizes)):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


def _scalar_type(dtype: str) -> int:
    try:
        return _SCALAR_TYPES[str(dtype)]
    except KeyError as error:
        raise BackendContractError("dtype {!r} has no Stable-ABI scalar code".format(dtype)) from error


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    name = str(dtype)
    if not name.startswith("torch.") or "." in name[6:]:
        raise BackendContractError("invalid PyTorch dtype name {!r}".format(dtype))
    value = getattr(torch_module, name[6:], None)
    if value is None:
        raise NativeTorchUnavailableError("dtype", 7, "PyTorch lacks {}".format(name))
    return value


def _resolved_wrapper(torch_module: Any) -> Any:
    try:
        return torch_module.ops.xvram_internal._wrap_resolved_v1
    except AttributeError as error:
        raise NativeTorchUnavailableError(
            "stable_bridge", 7, "xvram_internal::_wrap_resolved_v1 is not registered"
        ) from error


def _aten_overload(torch_module: Any, name: str, overload: str) -> Any:
    try:
        return getattr(getattr(torch_module.ops.aten, name), overload)
    except AttributeError as error:
        raise NativeTorchUnavailableError(
            "operator_lookup", 7, "required aten.{}.{} is unavailable".format(name, overload)
        ) from error


def _argument_value_names(value: Any) -> Iterable[str]:
    if isinstance(value, ValueReference):
        yield value.name
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _argument_value_names(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _argument_value_names(item)


def _resolve_argument(value: Any, views: Mapping[str, Any], torch_module: Any) -> Any:
    if isinstance(value, ValueReference):
        try:
            return views[value.name]
        except KeyError as error:
            raise BackendContractError("planned tensor view is unavailable") from error
    if isinstance(value, tuple):
        return tuple(_resolve_argument(item, views, torch_module) for item in value)
    if isinstance(value, list):
        return [_resolve_argument(item, views, torch_module) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _resolve_argument(item, views, torch_module) for key, item in value.items()
        }
    if isinstance(value, str) and value.startswith("torch."):
        name = value[6:]
        resolved = getattr(torch_module, name, None)
        if resolved is None:
            raise NativeTorchUnavailableError(
                "constant", 7, "PyTorch lacks graph constant {}".format(value)
            )
        return resolved
    return value


__all__ = [
    "NativeBackendFactory",
    "NativeErrorInfo",
    "NativeInferenceBackend",
    "NativeGemmToken",
    "NativeLeaseInfo",
    "NativeLeaseToken",
    "NativeRuntimeConfig",
    "NativeTorchRuntimeError",
    "NativeTorchUnavailableError",
    "TORCH_RUNTIME_ABI_VERSION",
    "XVRAM_TORCH_RUNTIME_ENV",
    "find_runtime_library",
]
