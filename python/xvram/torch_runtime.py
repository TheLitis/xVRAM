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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        *,
        prefetch_distance: Optional[int] = None,
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
        if prefetch_distance is None:
            backend_config = getattr(backend, "config", None)
            prefetch_distance = int(getattr(backend_config, "prefetch_distance", 2))
        if isinstance(prefetch_distance, bool) or not 0 <= int(prefetch_distance) <= 8:
            raise ValueError("prefetch_distance must be in [0, 8]")
        self._prefetch_distance = int(prefetch_distance)
        self._compute_nodes = tuple(
            node for node in self._plan.nodes if node.kind is NodeKind.COMPUTE
        )
        self._compute_positions = {
            node.index: position for position, node in enumerate(self._compute_nodes)
        }
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
        if self._prefetch_distance == 0:
            return ()
        ranges: List[ManagedRange] = []
        embedding_by_weight = {
            candidate.embedding_range.weight_value: candidate.embedding_range
            for candidate in self._plan.nodes
            if candidate.embedding_range is not None
        }
        try:
            position = self._compute_positions[node.index]
        except KeyError as error:
            raise BackendContractError("compute node is absent from the static schedule") from error
        current_inputs = set(node.input_values)
        future_reads: Dict[str, int] = {}
        future_nodes = self._compute_nodes[
            position + 1 : position + 1 + self._prefetch_distance
        ]
        for future_position, future in enumerate(future_nodes, start=position + 1):
            for name in future.input_values:
                candidate = self._plan.value(name)
                if candidate.is_persistent and name not in current_inputs:
                    future_reads.setdefault(name, future_position)
        ordered_names = (
            name
            for name, _position in sorted(
                future_reads.items(), key=lambda item: (item[1], item[0])
            )
        )
        for name in ordered_names:
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


def run_benchmark_plan(
    configuration: Mapping[str, Any],
    *,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
    trace: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """Execute one isolated Phase 4b acceptance plan and build its v1 report.

    Imports that can initialize PyTorch or CUDA stay inside this worker-only
    entry point.  The controller can therefore import :mod:`xvram.torch_runtime`
    while remaining completely CUDA-free.
    """

    import time

    from .torch_acceptance import (
        output_digest,
        output_error,
        prepare_meta_llama,
        run_layer_streamed_reference,
    )
    from .torch_native import (
        NativeInferenceBackend,
        NativeRuntimeConfig,
        NativeTorchUnavailableError,
    )
    from .torch_report import (
        EXIT_COMPLETED,
        EXIT_CORRUPTION,
        EXIT_RUNTIME,
        EXIT_USAGE,
        empty_report,
        finalize_proof,
        validate_report_envelope,
    )
    from .torch_workload import Llama2LikeConfig

    emit_progress = progress or (lambda _event: None)
    emit_trace_batch = trace or (lambda _records: None)
    config = dict(configuration)
    report = empty_report(
        config,
        exit_code=EXIT_RUNTIME,
        status="failed",
        message="inference worker failed",
        include_identifiers=bool(config.get("include_identifiers", False)),
    )
    # Validating the empty envelope also validates every untrusted plan field.
    try:
        validate_report_envelope(report)
    except (TypeError, ValueError) as error:
        report["outcome"].update(
            status="failed", exit_code=EXIT_USAGE, message=str(error)
        )
        report["diagnostics"].append(
            {"stage": "plan", "code": "invalid_worker_plan", "message": str(error)}
        )
        return finalize_proof(report)

    trace_sequence = 0

    def worker_event(
        *,
        kind: str,
        operation: str,
        bytes_value: int = 0,
        region_id: Optional[int] = None,
        reason: str = "",
        progress_fields: Optional[Mapping[str, Any]] = None,
    ) -> None:
        nonlocal trace_sequence
        payload: Dict[str, Any] = {
            "stage": kind,
            "operation": operation,
            "bytes": max(0, int(bytes_value)),
        }
        if region_id is not None:
            payload["region_id"] = int(region_id)
        if progress_fields:
            payload.update(dict(progress_fields))
        emit_progress(payload)
        trace_sequence += 1
        emit_trace_batch(
            [
                {
                    "schema_version": 1,
                    "record_type": "xvram.pytorch_trace",
                    "sequence": trace_sequence,
                    "monotonic_ns": time.monotonic_ns(),
                    "kind": kind,
                    "region_id": int(region_id) if region_id is not None else None,
                    "allocation_id": None,
                    "operation": operation,
                    "bytes": max(0, int(bytes_value)),
                    "reason": str(reason),
                }
            ]
        )

    def native_progress(event: Mapping[str, Any]) -> None:
        stage = str(event.get("stage", "transition"))
        kind = {
            "region": "lease",
            "prefetch": "prefetch",
            "discard": "discard",
            "scratch": "scratch",
        }.get(stage, "transition")
        region = event.get("region_id")
        worker_event(
            kind=kind,
            operation="{}_{}".format(stage, event.get("operation", "retired")),
            bytes_value=int(event.get("bytes", 0)),
            region_id=int(region) if isinstance(region, int) and region > 0 else None,
            reason=str(event.get("adapter", "runtime")),
            progress_fields={
                key: value
                for key, value in event.items()
                if key
                not in {
                    "stage",
                    "operation",
                    "bytes",
                    "storage_id",
                }
                and isinstance(value, (str, int, float, bool))
            },
        )

    backend: Optional[NativeInferenceBackend] = None
    runtime: Optional[InferenceRuntime] = None
    telemetry: Dict[str, Any] = {}
    output: Any = None
    primary_error: Optional[BaseException] = None
    cleanup_error: Optional[BaseException] = None
    persistent_read_only = False

    try:
        import torch

        if not bool(torch.cuda.is_available()):
            raise NativeTorchUnavailableError("cuda_preflight", 7, "PyTorch CUDA is unavailable")
        device = int(config["device"])
        if device >= int(torch.cuda.device_count()):
            raise NativeTorchUnavailableError(
                "cuda_preflight", 7, "requested CUDA device does not exist"
            )
        torch.cuda.set_device(device)
        torch.cuda.init()
        properties = torch.cuda.get_device_properties(device)
        report["build"].update(
            torch_version=str(torch.__version__),
            cuda_version=str(getattr(torch.version, "cuda", None) or "unavailable"),
        )
        report["device"].update(
            ordinal=device,
            name=(
                str(properties.name)
                if bool(config.get("include_identifiers", False))
                else "redacted"
            ),
            total_vram_bytes=int(properties.total_memory),
            compute_capability="{}.{}".format(
                int(properties.major), int(properties.minor)
            ),
            identifiers_included=bool(config.get("include_identifiers", False)),
        )
        worker_event(kind="transition", operation="cuda_preflight", reason="ready")

        dtype_name = str(config["dtype"])
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
        model_config = Llama2LikeConfig(
            layers=int(config["layers"]),
            hidden=int(config["hidden"]),
            intermediate=int(config["intermediate"]),
            heads=int(config["heads"]),
            batch=int(config["batch"]),
            sequence=int(config["sequence"]),
        )
        prepared = prepare_meta_llama(
            model_config,
            seed=int(str(config["seed"]), 16),
            dtype=dtype,
            chunk_bytes=int(config["chunk_size_bytes"]),
        )
        worker_event(kind="transition", operation="state_plan", reason="prepared")
        captured = capture_inference(
            prepared.module,
            prepared.example_inputs,
            state_provider=prepared.state_provider,
            torch_module=torch,
        )
        plan = captured.plan
        worker_event(kind="transition", operation="strict_export", reason="captured")

        persistent_storages = {
            value.storage_id for value in plan.values.values() if value.is_persistent
        }
        persistent_read_only = all(
            access.mode is AccessMode.READ
            for node in plan.nodes
            for access in node.accesses
            if access.storage_id in persistent_storages
        )
        if not persistent_read_only:
            raise PlanError("persistent model state contains a write-capable access")

        non_state_storages: Dict[str, int] = {}
        for value in plan.values.values():
            if not value.is_persistent:
                non_state_storages[value.storage_id] = int(value.storage_bytes)
        activation_bytes = sum(non_state_storages.values())
        if int(plan.logical_state_bytes) != int(prepared.state_provider.logical_state_bytes):
            raise PlanError("planner and streaming state byte counts do not reconcile")
        report["model"].update(
            kind=str(config["model"]),
            layers=model_config.layers,
            batch=model_config.batch,
            sequence=model_config.sequence,
            hidden=model_config.hidden,
            intermediate=model_config.intermediate,
            heads=model_config.heads,
            parameter_bytes=int(prepared.state_provider.parameter_bytes),
            activation_bytes=activation_bytes,
            logical_bytes=int(plan.logical_state_bytes) + activation_bytes,
        )
        region_count = sum(1 for node in plan.nodes if node.kind is NodeKind.COMPUTE)
        report["graph"].update(
            hash=plan.graph_hash,
            backend_hash=plan.graph_hash,
            allowlist_hash=plan.allowlist_hash,
            node_count=len(plan.nodes),
            region_count=region_count,
            unsupported_nodes=[],
        )

        native_config = NativeRuntimeConfig(
            device=device,
            policy=str(config["policy"]),
            chunk_size=int(config["chunk_size_bytes"]),
            cache_target=int(config["cache_target_bytes"]),
            device_headroom=int(config["device_headroom_bytes"]),
            scratch_cap=int(config["scratch_cap_bytes"]),
            prefetch_distance=int(config["prefetch_distance"]),
            sdpa_backend=str(config["sdpa_backend"]),
        )
        backend = NativeInferenceBackend(
            config=native_config,
            _torch_module=torch,
            _progress_callback=native_progress,
        )
        runtime = InferenceRuntime(plan, backend, prepared.state_provider)
        output = runtime.run(*prepared.example_inputs)
        if isinstance(output, tuple):
            raise BackendContractError("acceptance graph produced multiple outputs")
        output_hash = output_digest(output)
        worker_event(kind="verification", operation="managed_output", reason=output_hash)

        runtime.close()
        telemetry = backend.telemetry_snapshot()
        torch.cuda.empty_cache()

        def reference_progress(item: Any) -> None:
            worker_event(
                kind="verification",
                operation="reference_{}".format(str(item.stage).replace(":", "_")),
                region_id=None,
                reason="{}/{}".format(int(item.completed), int(item.total)),
            )

        reference = run_layer_streamed_reference(
            prepared,
            device="cuda:{}".format(device),
            sdpa_backend=str(config["sdpa_backend"]),
            progress=reference_progress,
        )
        comparison = output_error(output, reference.output)
        reference_hash = reference.digest
        exact_match = output_hash == reference_hash
        report["verification"].update(
            reference_digest=reference_hash,
            output_digest=output_hash,
            max_abs_error=_finite_report_number(comparison.max_absolute),
            max_rel_error=_finite_report_number(comparison.max_relative),
            mismatch_count=(
                int(comparison.mismatch_count)
                if exact_match
                else max(1, int(comparison.mismatch_count))
            ),
        )
        if exact_match and comparison.within_tolerance:
            report["outcome"].update(
                status="completed",
                exit_code=EXIT_COMPLETED,
                message="lease-scoped PyTorch inference proof completed",
            )
        else:
            report["outcome"].update(
                status="corruption",
                exit_code=EXIT_CORRUPTION,
                message="managed output differs from the streamed PyTorch reference",
            )
            report["diagnostics"].append(
                {
                    "stage": "verification",
                    "code": "reference_mismatch",
                    "message": "{} mismatched elements; exact digest match={}".format(
                        comparison.mismatch_count, exact_match
                    ),
                }
            )
    except BaseException as error:
        primary_error = error
    finally:
        if runtime is not None and runtime.state is not RuntimeState.CLOSED:
            try:
                runtime.close()
            except BaseException as error:
                cleanup_error = error
        elif backend is not None and not backend.close_succeeded:
            try:
                backend.close()
            except BaseException as error:
                cleanup_error = error
        if backend is not None:
            try:
                telemetry = backend.telemetry_snapshot()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

    if backend is not None:
        _apply_benchmark_telemetry(
            report,
            telemetry,
            close_succeeded=backend.close_succeeded,
            persistent_read_only=persistent_read_only,
        )
    report["execution"]["trace_records"] = trace_sequence

    failure = cleanup_error or primary_error
    if failure is not None:
        exit_code, status, code = _classify_benchmark_failure(failure)
        if cleanup_error is not None:
            exit_code, status, code = EXIT_RUNTIME, "failed", "cleanup_failure"
        report["outcome"].update(
            status=status, exit_code=exit_code, message=str(failure)
        )
        report["diagnostics"].append(
            {
                "stage": "cleanup" if cleanup_error is not None else "worker",
                "code": code,
                "message": str(failure),
            }
        )
    return finalize_proof(report)


def _finite_report_number(value: Any) -> float:
    import math
    import sys

    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0.0 else sys.float_info.max


def _classify_benchmark_failure(error: BaseException) -> Tuple[int, str, str]:
    from .torch_native import NativeTorchRuntimeError, NativeTorchUnavailableError
    from .torch_report import (
        EXIT_INTERNAL,
        EXIT_PRESSURE,
        EXIT_PREREQUISITE,
        EXIT_RUNTIME,
        EXIT_TIMEOUT,
    )

    if isinstance(error, NativeTorchUnavailableError):
        return EXIT_PREREQUISITE, "skipped", "prerequisite_unavailable"
    if isinstance(error, NativeTorchRuntimeError):
        if error.status in {8, 9, 10}:
            return EXIT_PRESSURE, "oom", "memory_pressure"
        if error.status == 11:
            return EXIT_TIMEOUT, "timeout", "runtime_timeout"
        if error.status in {1, 2, 3, 6, 7}:
            return EXIT_PREREQUISITE, "skipped", "prerequisite_unavailable"
        return EXIT_RUNTIME, "failed", "native_runtime_failure"
    if isinstance(error, (PlanError, BackendContractError, ValueError)):
        return EXIT_PREREQUISITE, "skipped", "unsupported_or_invalid_plan"
    if isinstance(error, MemoryError):
        return EXIT_PRESSURE, "oom", "host_out_of_memory"
    if isinstance(error, (AssertionError, TypeError, KeyError)):
        return EXIT_INTERNAL, "failed", "internal_error"
    return EXIT_RUNTIME, "failed", "runtime_failure"


def _apply_benchmark_telemetry(
    report: Dict[str, Any],
    telemetry: Mapping[str, Any],
    *,
    close_succeeded: bool,
    persistent_read_only: bool,
) -> None:
    value = lambda name, default=0: int(telemetry.get(name, default))
    gemm_tiles_submitted = value("gemm_tiles_submitted")
    gemm_tiles_completed = value("gemm_tiles_completed")
    leases_acquired = value("leases_acquired")
    leases_retired = value("leases_retired")
    events_recorded = value("events_recorded") + value("gemm_events_recorded")
    events_retired = value("events_retired") + value("gemm_events_retired")
    timings = [
        _finite_report_number(item)
        for item in telemetry.get("region_timings_ms", [])
    ]
    report["execution"].update(
        leases_acquired=leases_acquired,
        leases_sealed=value("leases_sealed"),
        leases_retired=leases_retired,
        tiled_gemm_regions=value("gemm_calls"),
        tiled_gemm_tiles=gemm_tiles_completed,
        events_recorded=events_recorded,
        events_retired=events_retired,
        live_views_peak=value("tensor_views_peak"),
        live_views_final=value("tensor_views_live"),
        regions_completed=value("nodes_retired"),
        region_timings_ms=timings,
        scratch_peak_bytes=value("scratch_bytes_peak"),
    )
    report["cache"].update(
        target_bytes=value("cache_target_bytes", report["cache"]["target_bytes"]),
        resident_bytes=value("resident_bytes"),
        resident_peak_bytes=value("resident_peak_bytes"),
        maps=value("maps"),
        set_access=value("set_access_calls"),
        unmaps=value("unmaps"),
        hits=value("cache_hits"),
        misses=value("cache_misses"),
        h2d_bytes=value("h2d_bytes"),
        d2h_bytes=value("d2h_bytes"),
        weight_d2h_bytes=0 if persistent_read_only else value("d2h_bytes"),
        evictions=value("clean_evictions") + value("dirty_evictions"),
        frame_reuses=value("handles_reused"),
        prefetch_submitted=value("prefetches"),
        prefetch_retired=value("prefetches") if close_succeeded else 0,
        unsafe_remaps=value("unsafe_remaps"),
        unsafe_transitions=value("unsafe_transitions"),
    )
    report["proof"]["stable_addresses"] = bool(
        telemetry.get("stable_addresses", False)
    ) and bool(telemetry.get("no_physical_aliases", False))
    allocations_reconciled = value("allocations_created") == value("allocations_released")
    mappings_reconciled = value("maps") == value("unmaps") and value("resident_bytes") == 0
    leases_reconciled = (
        leases_acquired == value("leases_sealed") == leases_retired
        and gemm_tiles_submitted == gemm_tiles_completed
        and events_recorded == events_retired
    )
    cleanup = report["cleanup"]
    cleanup.update(
        leases_drained=leases_reconciled,
        views_released=value("tensor_views_live") == 0,
        mappings_unmapped=mappings_reconciled,
        handles_released=bool(close_succeeded and mappings_reconciled),
        reservations_released=bool(close_succeeded and allocations_reconciled),
        streams_destroyed=bool(close_succeeded),
        context_released=bool(close_succeeded),
    )
    cleanup["complete"] = all(
        bool(item) for name, item in cleanup.items() if name != "complete"
    )


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
    "run_benchmark_plan",
]
