"""Public lease-scoped PyTorch inference frontend.

Importing this module neither imports PyTorch nor loads CUDA.  Strict export,
native library loading, and primary-context attachment happen only when a graph
is compiled.  Managed CUDA tensor views never cross this API boundary.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from .torch_native import NativeInferenceBackend, NativeRuntimeConfig
from .torch_planner import (
    InferencePlan,
    ManagedValue,
    PlanError,
    ValueReference,
    capture_inference,
)
from .torch_runtime import (
    BackendContractError,
    CpuModuleStateProvider,
    InferenceBackend,
    InferenceBackendFactory,
    InferenceRuntime as _Scheduler,
    RuntimeState,
    StateProvider,
)


class InferenceRuntimeError(RuntimeError):
    """The public inference runtime was used outside its static contract."""


@dataclass(frozen=True)
class CompileMetadata:
    graph_hash: str
    backend_hash: str
    allowlist_hash: str
    node_count: int
    region_count: int


class CompiledInference:
    """One strict static-shape graph bound to a native xVRAM session."""

    def __init__(
        self,
        owner: "InferenceRuntime",
        scheduler: _Scheduler,
        backend: InferenceBackend,
        exported_program: Any,
    ) -> None:
        self._owner = owner
        self._scheduler = scheduler
        self._backend = backend
        self._exported_program = exported_program
        plan = scheduler.plan
        regions = sum(1 for node in plan.nodes if node.kind.value == "compute")
        self._metadata = CompileMetadata(
            graph_hash=plan.graph_hash,
            backend_hash=plan.graph_hash,
            allowlist_hash=plan.allowlist_hash,
            node_count=len(plan.nodes),
            region_count=regions,
        )

    @property
    def plan(self) -> InferencePlan:
        return self._scheduler.plan

    @property
    def exported_program(self) -> Any:
        return self._exported_program

    @property
    def metadata(self) -> CompileMetadata:
        return self._metadata

    @property
    def state(self) -> RuntimeState:
        return self._scheduler.state

    def __call__(self, *cpu_inputs: Any) -> Any:
        self._owner._require_active(self)
        return self._scheduler.run(*cpu_inputs)

    def telemetry(self) -> Mapping[str, Union[int, float, bool]]:
        snapshot = getattr(self._backend, "telemetry_snapshot", None)
        if snapshot is None or not callable(snapshot):
            return {}
        return dict(snapshot())

    def materialize_cuda(self, *cpu_inputs: Any) -> Any:
        """Run inference and explicitly copy result(s) into PyTorch-owned CUDA memory.

        The copy is permitted only when its complete output fits the configured
        device-headroom reserve.  It never exposes or transfers ownership of a
        managed xVRAM address.
        """

        host_output = self(*cpu_inputs)
        outputs = host_output if isinstance(host_output, tuple) else (host_output,)
        total_bytes = 0
        for value in outputs:
            numel = getattr(value, "numel", None)
            element_size = getattr(value, "element_size", None)
            if not callable(numel) or not callable(element_size):
                raise InferenceRuntimeError("material output is not tensor-like")
            total_bytes += int(numel()) * int(element_size())
        if total_bytes > self._owner.config.headroom_bytes:
            raise InferenceRuntimeError(
                "CUDA materialization exceeds the reserved device headroom"
            )
        torch_module = self._owner._torch_module or importlib.import_module("torch")
        device = "cuda:{}".format(self._owner.config.device)
        materialized = tuple(value.to(device=device, non_blocking=False) for value in outputs)
        return materialized[0] if len(materialized) == 1 else materialized

    def close(self) -> None:
        self._owner._close_compiled(self)


class InferenceRuntime:
    """Own one single-GPU, static-shape, forward-inference session.

    A runtime intentionally accepts one compiled graph.  This mirrors the
    native quarantine boundary and guarantees that the runtime-owned CUDA
    stream and every ephemeral view have a single unambiguous lifetime.
    """

    def __init__(
        self,
        *,
        device: int = 0,
        cache_target: Union[int, str] = "auto",
        chunk_size: Union[int, str] = "64MiB",
        device_headroom: Union[int, str] = "512MiB",
        scratch_cap: Union[int, str] = "512MiB",
        compression: str = "off",
        compression_codec: str = "auto",
        host_store_cap: Union[int, str] = "auto",
        host_headroom: Union[int, str] = "auto",
        compression_scratch_cap: Union[int, str] = "256MiB",
        codec_slots: int = 2,
        codec_workers: int = 2,
        staging_slots: int = 4,
        policy: str = "clock",
        prefetch_distance: int = 2,
        sdpa_backend: str = "math",
        stall_timeout_ms: int = 5000,
        budget_poll_ms: int = 100,
        maximum_region_ms: int = 250,
        library_path: Optional[Union[str, PathLike[str]]] = None,
        _torch_module: Any = None,
        _backend_factory: Any = None,
    ) -> None:
        self.config = NativeRuntimeConfig(
            device=device,
            policy=policy,
            chunk_size=chunk_size,
            cache_target=cache_target,
            device_headroom=device_headroom,
            scratch_cap=scratch_cap,
            compression=compression,
            compression_codec=compression_codec,
            host_store_cap=host_store_cap,
            host_headroom=host_headroom,
            compression_scratch_cap=compression_scratch_cap,
            codec_slots=codec_slots,
            codec_workers=codec_workers,
            staging_slots=staging_slots,
            stall_timeout_ms=stall_timeout_ms,
            budget_poll_ms=budget_poll_ms,
            maximum_region_ms=maximum_region_ms,
            prefetch_distance=prefetch_distance,
            sdpa_backend=sdpa_backend,
        ).validate()
        self._library_path = library_path
        self._torch_module = _torch_module
        self._backend_factory = _backend_factory
        self._compiled: Optional[CompiledInference] = None
        self._closed = False
        self._owner_thread = 0
        self._lock = threading.RLock()

    def compile_inference(
        self,
        model: Any,
        example_inputs: Sequence[Any],
        *,
        example_kwargs: Optional[Mapping[str, Any]] = None,
        state_provider: Optional[StateProvider] = None,
    ) -> CompiledInference:
        """Strictly export, preflight, allocate, and load one inference graph."""

        examples = _example_tuple(example_inputs)
        with self._lock:
            self._require_owner_or_bind()
            if self._closed:
                raise InferenceRuntimeError("runtime is closed")
            if self._compiled is not None:
                raise InferenceRuntimeError("runtime already owns a compiled graph")
            provider = state_provider or CpuModuleStateProvider(model)
            if not isinstance(provider, StateProvider):
                raise TypeError("state_provider must implement StateProvider")
            captured = capture_inference(
                model,
                examples,
                example_kwargs,
                state_provider=provider,
                torch_module=self._torch_module,
            )
            backend = self._create_backend(captured.plan)
            scheduler = _Scheduler(
                captured.plan,
                backend,
                provider,
                prefetch_distance=self.config.prefetch_distance,
            )
            compiled = CompiledInference(
                self, scheduler, backend, captured.exported_program
            )
            try:
                scheduler.prepare()
            except BaseException:
                try:
                    scheduler.close()
                except BaseException as cleanup:
                    raise InferenceRuntimeError(
                        "graph preparation failed and native cleanup also failed"
                    ) from cleanup
                raise
            self._compiled = compiled
            return compiled

    def backend(
        self,
        model: Any,
        example_inputs: Sequence[Any],
        *,
        example_kwargs: Optional[Mapping[str, Any]] = None,
        state_provider: Optional[StateProvider] = None,
    ) -> Callable[[Any, Sequence[Any]], Callable[..., Any]]:
        """Return a strict custom backend for ``torch.compile``.

        The canonical graph is always captured first with strict
        :func:`torch.export.export`.  Dynamo's graph and lifted state are then
        validated against that exact capture before the already-compiled xVRAM
        callable is returned.  A graph break or structural mismatch raises;
        there is no eager fallback.
        """

        examples = _example_tuple(example_inputs)
        provider = state_provider or CpuModuleStateProvider(model)
        compiled = self.compile_inference(
            model,
            examples,
            example_kwargs=example_kwargs,
            state_provider=provider,
        )
        return _TorchCompileBackend(compiled, provider, examples)

    def close(self) -> None:
        with self._lock:
            self._require_owner_or_bind()
            if self._closed:
                return
            compiled = self._compiled
            self._compiled = None
            self._closed = True
        if compiled is not None:
            compiled._scheduler.close()

    def __enter__(self) -> "InferenceRuntime":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _create_backend(self, plan: InferencePlan) -> InferenceBackend:
        factory = self._backend_factory
        if factory is None:
            return NativeInferenceBackend(
                config=self.config,
                library_path=self._library_path,
                _torch_module=self._torch_module,
            )
        if isinstance(factory, InferenceBackendFactory):
            backend = factory.create(plan)
        elif callable(factory):
            backend = factory(plan)
        else:
            raise TypeError("_backend_factory must be callable or InferenceBackendFactory")
        if not isinstance(backend, InferenceBackend):
            raise TypeError("backend factory returned an invalid backend")
        return backend

    def _require_active(self, compiled: CompiledInference) -> None:
        with self._lock:
            self._require_owner_or_bind()
            if self._closed or self._compiled is not compiled:
                raise InferenceRuntimeError("compiled inference is no longer active")

    def _close_compiled(self, compiled: CompiledInference) -> None:
        with self._lock:
            self._require_owner_or_bind()
            if self._compiled is not compiled:
                if self._closed:
                    return
                raise InferenceRuntimeError("compiled inference belongs to another runtime")
        self.close()

    def _require_owner_or_bind(self) -> None:
        current = threading.get_ident()
        if self._owner_thread == 0:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise InferenceRuntimeError("runtime must remain on its owning thread")


class _TorchCompileBackend:
    def __init__(
        self,
        compiled: CompiledInference,
        state_provider: StateProvider,
        example_inputs: Tuple[Any, ...],
    ) -> None:
        self.compiled = compiled
        self.state_provider = state_provider
        self.example_inputs = example_inputs
        self.graph_hash = compiled.plan.graph_hash
        self.backend_hash = ""

    def __call__(self, graph_module: Any, callback_inputs: Sequence[Any]) -> Callable[..., Any]:
        _validate_dynamo_graph(graph_module, self.compiled.plan)
        positions = _locate_user_inputs(
            tuple(callback_inputs), self.example_inputs, self.state_provider
        )
        sequence_output = _dynamo_sequence_output(graph_module)
        self.backend_hash = self.graph_hash

        def execute(*arguments: Any) -> Any:
            if len(arguments) != len(callback_inputs):
                raise BackendContractError("torch.compile callback arity changed")
            result = self.compiled(*(arguments[index] for index in positions))
            if not sequence_output:
                return result
            if len(self.compiled.plan.output_names) == 1:
                return (result,)
            return tuple(result)

        return execute


_DYNAMO_FUNCTION_TARGETS = {
    "embedding": "aten.embedding.default",
    "linear": "aten.linear.default",
    "rms_norm": "aten.rms_norm.default",
    "scaled_dot_product_attention": "aten.scaled_dot_product_attention.default",
    "silu": "aten.silu.default",
    "add": "aten.add.Tensor",
    "mul": "aten.mul.Tensor",
    "neg": "aten.neg.default",
    "cat": "aten.cat.default",
    "mm": "aten.mm.default",
    "addmm": "aten.addmm.default",
    "bmm": "aten.bmm.default",
    "clone": "aten.clone.default",
    "_to_copy": "aten._to_copy.default",
}
_DYNAMO_METHOD_TARGETS = {
    "view": "aten.view.default",
    "reshape": "aten.reshape.default",
    "transpose": "aten.transpose.int",
    "permute": "aten.permute.default",
    "unsqueeze": "aten.unsqueeze.default",
    "slice": "aten.slice.Tensor",
}


def _validate_dynamo_graph(graph_module: Any, plan: InferencePlan) -> None:
    graph = getattr(graph_module, "graph", None)
    nodes = list(getattr(graph, "nodes", ()) or ())
    if not nodes:
        raise BackendContractError("torch.compile backend received an empty graph")
    actual = []
    for node in nodes:
        op = str(getattr(node, "op", ""))
        if op not in {"call_function", "call_method"}:
            continue
        target = _normalize_dynamo_target(node, op)
        meta = getattr(node, "meta", {}) or {}
        value = meta.get("example_value", meta.get("val"))
        sizes = tuple(int(item) for item in getattr(value, "shape", ()) or ())
        dtype = str(getattr(value, "dtype", ""))
        actual.append((target, sizes, dtype))

    expected = []
    for node in plan.nodes:
        value = plan.value(node.output_value)
        expected.append((node.target, value.spec.sizes, value.spec.dtype))
    if actual != expected:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(actual, expected))
                if pair[0] != pair[1]
            ),
            min(len(actual), len(expected)),
        )
        raise BackendContractError(
            "torch.compile graph differs from strict export at node {} "
            "(dynamo_nodes={}, export_nodes={})".format(
                mismatch, len(actual), len(expected)
            )
        )


def _dynamo_sequence_output(graph_module: Any) -> bool:
    graph = getattr(graph_module, "graph", None)
    for node in getattr(graph, "nodes", ()) or ():
        if str(getattr(node, "op", "")) != "output":
            continue
        args = tuple(getattr(node, "args", ()) or ())
        return bool(args) and isinstance(args[0], (tuple, list))
    raise BackendContractError("torch.compile graph has no output node")


def _normalize_dynamo_target(node: Any, op: str) -> str:
    target = getattr(node, "target", None)
    if op == "call_method":
        name = str(target)
        try:
            return _DYNAMO_METHOD_TARGETS[name]
        except KeyError as error:
            raise BackendContractError(
                "torch.compile produced unsupported method {!r}".format(name)
            ) from error
    name = getattr(target, "__name__", None) or str(target).rsplit(".", 1)[-1]
    if name == "getitem":
        return "aten.slice.Tensor" if _contains_slice(getattr(node, "args", ())) else "operator.getitem"
    try:
        return _DYNAMO_FUNCTION_TARGETS[str(name)]
    except KeyError as error:
        raise BackendContractError(
            "torch.compile produced unsupported function {!r}".format(name)
        ) from error


def _contains_slice(value: Any) -> bool:
    if isinstance(value, slice):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_slice(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_slice(item) for item in value.values())
    return False


def _locate_user_inputs(
    callback_inputs: Tuple[Any, ...],
    examples: Tuple[Any, ...],
    provider: StateProvider,
) -> Tuple[int, ...]:
    state_identities = {
        identity
        for identity in (_tensor_identity(provider.describe(name).tensor) for name in provider.targets())
        if identity is not None
    }
    positions = []
    used = set()
    for example in examples:
        identity = _tensor_identity(example)
        matches = [
            index
            for index, candidate in enumerate(callback_inputs)
            if index not in used
            and (candidate is example or (identity is not None and _tensor_identity(candidate) == identity))
        ]
        if len(matches) != 1:
            raise BackendContractError(
                "torch.compile did not preserve one unambiguous static user input"
            )
        index = matches[0]
        if identity is not None and identity in state_identities:
            raise BackendContractError("user input aliases persistent model state")
        positions.append(index)
        used.add(index)
    for index, candidate in enumerate(callback_inputs):
        if index in used:
            continue
        identity = _tensor_identity(candidate)
        if identity is None or identity not in state_identities:
            raise BackendContractError(
                "torch.compile lifted an input outside the canonical model state"
            )
    return tuple(positions)


def _tensor_identity(value: Any) -> Optional[Tuple[int, int]]:
    storage_method = getattr(value, "untyped_storage", None)
    if storage_method is None or not callable(storage_method):
        return None
    try:
        storage = storage_method()
        return int(storage.data_ptr()), int(storage.nbytes())
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        return None


def _example_tuple(example_inputs: Sequence[Any]) -> Tuple[Any, ...]:
    if isinstance(example_inputs, (str, bytes)) or not isinstance(example_inputs, Sequence):
        raise TypeError("example_inputs must be a sequence")
    return tuple(example_inputs)


__all__ = [
    "CompileMetadata",
    "CompiledInference",
    "InferenceRuntime",
    "InferenceRuntimeError",
    "PlanError",
]
