"""Execution-front-end contracts for graph-managed PyTorch inference.

This module intentionally contains no CUDA or native-library calls.  The
event-safe bridge is represented by :class:`InferenceBackend`, allowing the
planner and scheduler to be tested with a recording fake while a native
implementation owns CUDA context, VMM, stream, and event details.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .torch_planner import (
    AccessMode,
    InferencePlan,
    ManagedRange,
    ManagedValue,
    NodeKind,
    PlanError,
    PlannedNode,
    StateTensorInfo,
    build_inference_plan,
    capture_inference,
    describe_cpu_state_tensor,
    embedding_row_ranges,
    merge_managed_ranges,
)


class InferenceRuntimeError(RuntimeError):
    """Base error for the managed inference scheduler."""


class BackendContractError(InferenceRuntimeError):
    """A backend violated the stream/event-safe execution contract."""


class RuntimeState(str, Enum):
    NEW = "new"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    CLOSED = "closed"


class StateProvider(ABC):
    """Describe named, host-resident model state without copying it."""

    @abstractmethod
    def describe(self, target: str) -> StateTensorInfo:
        """Return one immutable CPU storage descriptor."""

    @abstractmethod
    def targets(self) -> Tuple[str, ...]:
        """Return all available targets in deterministic order."""


class CpuModuleStateProvider(StateProvider):
    """Expose registered CPU parameters and buffers from a module.

    Duplicate names are retained during traversal so tied parameters resolve to
    the same :class:`StateTensorInfo.storage_key`.  No tensor is cloned, pinned,
    or moved.
    """

    def __init__(self, module: Any) -> None:
        self._tensors: Dict[str, Any] = {}
        self._collect(module, "named_parameters")
        self._collect(module, "named_buffers")
        if not self._tensors:
            # Parameterless exported models remain valid; a missing target will
            # still produce a precise error from describe().
            self._tensors = {}

    def _collect(self, module: Any, method_name: str) -> None:
        method = getattr(module, method_name, None)
        if method is None or not callable(method):
            raise TypeError("module does not provide callable {}()".format(method_name))
        try:
            items = method(recurse=True, remove_duplicate=False)
        except TypeError:
            try:
                items = method(recurse=True)
            except TypeError:
                items = method()
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("{}() must yield (name, tensor) pairs".format(method_name))
            name, tensor = item
            if not isinstance(name, str) or not name:
                raise ValueError("state target names must be non-empty strings")
            previous = self._tensors.get(name)
            if previous is not None and previous is not tensor:
                raise ValueError("duplicate module state target {!r}".format(name))
            self._tensors[name] = tensor

    def describe(self, target: str) -> StateTensorInfo:
        try:
            tensor = self._tensors[target]
        except KeyError as error:
            raise PlanError("module state target {!r} is unavailable".format(target)) from error
        return describe_cpu_state_tensor(target, tensor)

    def targets(self) -> Tuple[str, ...]:
        return tuple(sorted(self._tensors))


@dataclass(frozen=True)
class ExecutionRequest:
    sequence: int
    node: PlannedNode
    ranges: Tuple[ManagedRange, ...]


class InferenceBackend(ABC):
    """Native/fake boundary for one event-safe managed inference session.

    Required ordering for a compute request is ``prefetch`` -> ``submit`` ->
    ``retire`` -> ``release_values``.  ``submit`` must not launch until every
    declared range is resident and pinned.  It must enqueue work on the runtime
    stream and establish a completion event generation.  ``retire`` returns
    only after that generation has completed successfully; consequently no
    release or remap is authorized before it returns.
    """

    @abstractmethod
    def prepare(self, plan: InferencePlan, state_provider: StateProvider) -> None:
        """Allocate logical storages and materialize aliases, without launches."""

    @abstractmethod
    def bind_input(self, value: ManagedValue, host_tensor: Any) -> None:
        """Copy or bind one validated pageable CPU input."""

    @abstractmethod
    def prefetch(self, ranges: Tuple[ManagedRange, ...]) -> None:
        """Submit advisory reads; an empty tuple is never passed."""

    @abstractmethod
    def submit(self, request: ExecutionRequest) -> Any:
        """Submit one planned compute node and return a non-null generation token."""

    @abstractmethod
    def retire(self, token: Any) -> None:
        """Establish successful completion of the exact submitted generation."""

    @abstractmethod
    def release_values(self, names: Tuple[str, ...]) -> None:
        """Apply post-retirement logical liveness hints."""

    @abstractmethod
    def read_output(self, value: ManagedValue) -> Any:
        """Return one host-visible output after all compute has retired."""

    @abstractmethod
    def close(self) -> None:
        """Drain and clean up; safe to call once after prepare()."""


class InferenceBackendFactory(ABC):
    """Create a backend for a hash-stable, fully validated plan."""

    @abstractmethod
    def create(self, plan: InferencePlan) -> InferenceBackend:
        """Return a fresh backend; implementations may cache by graph_hash."""


class InferenceRuntime:
    """Deterministic, single-transaction-at-a-time inference scheduler."""

    def __init__(
        self,
        plan: InferencePlan,
        backend: InferenceBackend,
        state_provider: StateProvider,
    ) -> None:
        if not isinstance(plan, InferencePlan):
            raise TypeError("plan must be an InferencePlan")
        if not isinstance(backend, InferenceBackend):
            raise TypeError("backend must implement InferenceBackend")
        if not isinstance(state_provider, StateProvider):
            raise TypeError("state_provider must implement StateProvider")
        self._plan = plan
        self._backend = backend
        self._state_provider = state_provider
        self._state = RuntimeState.NEW
        self._lock = threading.RLock()
        self._run_count = 0
        self._last_error: Optional[BaseException] = None
        self._backend_started = False

    @property
    def plan(self) -> InferencePlan:
        return self._plan

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._last_error

    def prepare(self) -> None:
        with self._lock:
            if self._state is RuntimeState.CLOSED:
                raise InferenceRuntimeError("runtime is closed")
            if self._state is RuntimeState.FAILED:
                raise InferenceRuntimeError("runtime is poisoned by an earlier failure")
            if self._state is RuntimeState.READY:
                return
            self._backend_started = True
            try:
                self._backend.prepare(self._plan, self._state_provider)
            except BaseException as error:
                self._last_error = error
                self._state = RuntimeState.FAILED
                raise
            self._state = RuntimeState.READY

    def run(self, *inputs: Any) -> Any:
        """Execute one static-shape inference and return host-visible output(s)."""

        with self._lock:
            self.prepare()
            if self._state is not RuntimeState.READY:
                raise InferenceRuntimeError("runtime is not ready")
            if len(inputs) != len(self._plan.input_names):
                raise ValueError(
                    "expected {} user inputs, received {}".format(
                        len(self._plan.input_names), len(inputs)
                    )
                )
            host_inputs: Dict[str, Any] = {}
            try:
                for name, tensor in zip(self._plan.input_names, inputs):
                    value = self._plan.value(name)
                    _validate_runtime_input(value, tensor)
                    self._backend.bind_input(value, tensor)
                    host_inputs[name] = tensor
                self._state = RuntimeState.RUNNING
                sequence = 0
                for node in self._plan.nodes:
                    if node.kind is NodeKind.ALIAS:
                        if node.release_values:
                            self._backend.release_values(node.release_values)
                        continue
                    ranges = list(node.accesses)
                    if node.embedding_range is not None:
                        template = node.embedding_range
                        tensor = host_inputs.get(template.indices_value)
                        if tensor is None:
                            raise BackendContractError(
                                "embedding indices are not a bound host input"
                            )
                        ranges.extend(
                            embedding_row_ranges(
                                _host_index_values(tensor),
                                vocabulary_rows=template.vocabulary_rows,
                                row_bytes=template.row_bytes,
                                storage_id=template.weight_storage_id,
                                storage_offset_bytes=template.weight_storage_offset_bytes,
                                value_name=template.weight_value,
                            )
                        )
                    merged = merge_managed_ranges(ranges)
                    prefetch = self._prefetch_ranges(node, host_inputs)
                    if prefetch:
                        self._backend.prefetch(prefetch)
                    sequence += 1
                    token = self._backend.submit(
                        ExecutionRequest(sequence=sequence, node=node, ranges=merged)
                    )
                    if token is None:
                        raise BackendContractError("backend returned a null generation token")
                    self._backend.retire(token)
                    if node.release_values:
                        self._backend.release_values(node.release_values)
                outputs = tuple(
                    self._backend.read_output(self._plan.value(name))
                    for name in self._plan.output_names
                )
                self._run_count += 1
                self._state = RuntimeState.READY
                return outputs[0] if len(outputs) == 1 else outputs
            except BaseException as error:
                self._last_error = error
                self._state = RuntimeState.FAILED
                raise

    def _prefetch_ranges(
        self, node: PlannedNode, host_inputs: Mapping[str, Any]
    ) -> Tuple[ManagedRange, ...]:
        ranges: List[ManagedRange] = []
        embedding_by_weight = {
            candidate.embedding_range.weight_value: candidate.embedding_range
            for candidate in self._plan.nodes
            if candidate.embedding_range is not None
        }
        for name in node.prefetch_values:
            template = embedding_by_weight.get(name)
            if template is not None and template.indices_value in host_inputs:
                ranges.extend(
                    embedding_row_ranges(
                        _host_index_values(host_inputs[template.indices_value]),
                        vocabulary_rows=template.vocabulary_rows,
                        row_bytes=template.row_bytes,
                        storage_id=template.weight_storage_id,
                        storage_offset_bytes=template.weight_storage_offset_bytes,
                        value_name=template.weight_value,
                    )
                )
                continue
            value = self._plan.value(name)
            ranges.append(
                ManagedRange(
                    storage_id=value.storage_id,
                    offset_bytes=value.storage_offset_bytes,
                    length_bytes=value.spec.span_bytes,
                    mode=AccessMode.READ,
                    values=(value.name,),
                )
            )
        return merge_managed_ranges(ranges) if ranges else ()

    def close(self) -> None:
        with self._lock:
            if self._state is RuntimeState.CLOSED:
                return
            try:
                if self._backend_started:
                    self._backend.close()
            except BaseException as error:
                self._last_error = error
                self._state = RuntimeState.FAILED
                raise
            self._state = RuntimeState.CLOSED

    def __enter__(self) -> "InferenceRuntime":
        self.prepare()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()


def compile_inference(
    module: Any,
    example_args: Sequence[Any],
    *,
    backend: Optional[InferenceBackend] = None,
    backend_factory: Optional[InferenceBackendFactory] = None,
    example_kwargs: Optional[Mapping[str, Any]] = None,
    state_provider: Optional[StateProvider] = None,
    torch_module: Any = None,
) -> InferenceRuntime:
    """Strictly export and compile a module for the managed scheduler."""

    provider = state_provider if state_provider is not None else CpuModuleStateProvider(module)
    captured = capture_inference(
        module,
        example_args,
        example_kwargs,
        state_provider=provider,
        torch_module=torch_module,
    )
    selected = _select_backend(captured.plan, backend, backend_factory)
    return InferenceRuntime(captured.plan, selected, provider)


def compile_exported_inference(
    exported_program: Any,
    *,
    backend: Optional[InferenceBackend] = None,
    backend_factory: Optional[InferenceBackendFactory] = None,
    state_provider: StateProvider,
) -> InferenceRuntime:
    """Compile an already-created strict static ExportedProgram."""

    plan = build_inference_plan(exported_program, state_provider=state_provider)
    selected = _select_backend(plan, backend, backend_factory)
    return InferenceRuntime(plan, selected, state_provider)


def _select_backend(
    plan: InferencePlan,
    backend: Optional[InferenceBackend],
    backend_factory: Optional[InferenceBackendFactory],
) -> InferenceBackend:
    if (backend is None) == (backend_factory is None):
        raise TypeError("provide exactly one of backend or backend_factory")
    if backend_factory is not None:
        if not isinstance(backend_factory, InferenceBackendFactory):
            raise TypeError("backend_factory must implement InferenceBackendFactory")
        backend = backend_factory.create(plan)
    if not isinstance(backend, InferenceBackend):
        raise TypeError("backend factory returned an invalid backend")
    return backend


def _validate_runtime_input(value: ManagedValue, tensor: Any) -> None:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type) != "cpu":
        raise ValueError("managed inference inputs must reside in pageable CPU memory")
    size_method = getattr(tensor, "size", None)
    stride_method = getattr(tensor, "stride", None)
    element_method = getattr(tensor, "element_size", None)
    if not callable(size_method) or not callable(stride_method) or not callable(element_method):
        raise TypeError("managed inference inputs must be tensor-like")
    sizes = tuple(int(item) for item in size_method())
    strides = tuple(int(item) for item in stride_method())
    dtype = str(getattr(tensor, "dtype", ""))
    element_size = int(element_method())
    expected = value.spec
    if (dtype, element_size, sizes, strides) != (
        expected.dtype,
        expected.element_size,
        expected.sizes,
        expected.strides,
    ):
        raise ValueError("input {!r} does not match its static export metadata".format(value.name))


def _host_index_values(tensor: Any) -> Tuple[int, ...]:
    detach = getattr(tensor, "detach", None)
    value = detach() if callable(detach) else tensor
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        value = reshape(-1)
    else:
        flatten = getattr(value, "flatten", None)
        value = flatten() if callable(flatten) else value
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else list(value)
    return tuple(int(item) for item in _flatten(raw))


def _flatten(values: Any) -> Iterable[Any]:
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


__all__ = [
    "BackendContractError",
    "CpuModuleStateProvider",
    "ExecutionRequest",
    "InferenceBackend",
    "InferenceBackendFactory",
    "InferenceRuntime",
    "InferenceRuntimeError",
    "RuntimeState",
    "StateProvider",
    "compile_exported_inference",
    "compile_inference",
]
