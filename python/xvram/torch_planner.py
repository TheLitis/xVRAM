"""Strict, side-effect-free planning for managed PyTorch inference.

The planner consumes a static ``torch.export.ExportedProgram`` but deliberately
does not import PyTorch at module import time.  This keeps the package usable on
controller and CI hosts without PyTorch while making the native execution
boundary explicit: every accepted tensor value, alias, access range, and
release point is present in :class:`InferencePlan`.

No unsupported operator is ever treated as an eager fallback.  Extending the
allowlist requires a version bump and a backend adapter with an explicit output
contract.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import operator
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple


PLANNER_VERSION = 1
OPERATOR_ALLOWLIST_VERSION = "xvram.torch_inference_ops_v1"


class PlanError(ValueError):
    """The exported graph cannot be executed by the strict managed backend."""

    def __init__(self, reason: str, *, node: str = "", target: str = "") -> None:
        parts = []
        if node:
            parts.append("node={!r}".format(node))
        if target:
            parts.append("target={!r}".format(target))
        detail = " ({})".format(", ".join(parts)) if parts else ""
        super().__init__(reason + detail)
        self.reason = reason
        self.node = node
        self.target = target


class ValueKind(str, Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    INPUT = "input"
    ACTIVATION = "activation"


class AccessMode(str, Enum):
    READ = "read"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"


class NodeKind(str, Enum):
    ALIAS = "alias"
    COMPUTE = "compute"


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    element_size: int
    sizes: Tuple[int, ...]
    strides: Tuple[int, ...]
    storage_offset_elements: int
    span_bytes: int
    numel: int


@dataclass(frozen=True)
class StateTensorInfo:
    """CPU state storage identity supplied by a state provider.

    ``storage_key`` may contain a process-local address.  It is used only to
    detect tied storage and is never serialized into the graph hash.
    """

    target: str
    tensor: Any
    storage_key: Hashable
    storage_bytes: int
    offset_bytes: int
    spec: TensorSpec


@dataclass(frozen=True)
class ManagedValue:
    name: str
    kind: ValueKind
    spec: TensorSpec
    producer_index: int
    first_use_index: int
    last_use_index: int
    storage_id: str
    storage_bytes: int
    storage_offset_bytes: int = 0
    alias_of: Optional[str] = None
    state_target: Optional[str] = None
    activation_slot: Optional[int] = None
    escapes_graph: bool = False

    @property
    def is_alias(self) -> bool:
        return self.alias_of is not None

    @property
    def is_persistent(self) -> bool:
        return self.kind in (ValueKind.PARAMETER, ValueKind.BUFFER, ValueKind.CONSTANT)


@dataclass(frozen=True)
class ManagedRange:
    storage_id: str
    offset_bytes: int
    length_bytes: int
    mode: AccessMode
    values: Tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingRange:
    weight_value: str
    indices_value: str
    vocabulary_rows: int
    row_bytes: int
    weight_storage_id: str
    weight_storage_offset_bytes: int


@dataclass(frozen=True)
class ActivationSlot:
    index: int
    storage_id: str
    capacity_bytes: int
    values: Tuple[str, ...]


@dataclass(frozen=True)
class ValueReference:
    name: str


@dataclass(frozen=True)
class PlannedNode:
    index: int
    name: str
    kind: NodeKind
    target: str
    adapter: str
    backend_target: str
    args: Any
    kwargs: Mapping[str, Any]
    input_values: Tuple[str, ...]
    output_value: str
    accesses: Tuple[ManagedRange, ...]
    embedding_range: Optional[EmbeddingRange]
    release_values: Tuple[str, ...]
    prefetch_values: Tuple[str, ...]


@dataclass(frozen=True)
class InferencePlan:
    version: int
    graph_hash: str
    allowlist_hash: str
    values: Mapping[str, ManagedValue]
    nodes: Tuple[PlannedNode, ...]
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    state_targets: Mapping[str, str]
    activation_slots: Tuple[ActivationSlot, ...]
    logical_state_bytes: int
    logical_activation_bytes: int
    peak_live_activation_bytes: int
    no_fallback: bool = True

    def value(self, name: str) -> ManagedValue:
        try:
            return self.values[name]
        except KeyError as error:
            raise KeyError("unknown managed value {!r}".format(name)) from error

    def is_hash_equivalent(self, other: object) -> bool:
        """Return whether another plan has the identical executable contract."""

        return (
            isinstance(other, InferencePlan)
            and self.version == other.version
            and self.allowlist_hash == other.allowlist_hash
            and self.graph_hash == other.graph_hash
        )


@dataclass(frozen=True)
class CapturedInference:
    exported_program: Any
    plan: InferencePlan


_COMPUTE_ADAPTERS: Mapping[str, str] = MappingProxyType(
    {
        "aten.embedding.default": "embedding_out",
        "aten.linear.default": "linear_out",
        "aten.rms_norm.default": "rms_norm_scratch_copy",
        "aten.scaled_dot_product_attention.default": "sdpa_math_scratch_copy",
        "aten.silu.default": "silu_out",
        "aten.add.Tensor": "add_out",
        "aten.mul.Tensor": "mul_out",
        "aten.neg.default": "neg_out",
        "aten.cat.default": "cat_out",
        "aten.mm.default": "mm_out",
        "aten.addmm.default": "addmm_out",
        "aten.bmm.default": "bmm_out",
        "aten.clone.default": "copy_out",
        "aten._to_copy.default": "cast_copy_out",
    }
)

_BACKEND_TARGETS: Mapping[str, str] = MappingProxyType(
    {
        "embedding_out": "aten.embedding.out",
        "linear_out": "aten.linear.out",
        "silu_out": "aten.silu.out",
        "add_out": "aten.add.out",
        "mul_out": "aten.mul.out",
        "neg_out": "aten.neg.out",
        "cat_out": "aten.cat.out",
        "mm_out": "aten.mm.out",
        "addmm_out": "aten.addmm.out",
        "bmm_out": "aten.bmm.out",
        "copy_out": "aten.copy.out",
        "cast_copy_out": "aten.copy.out",
        "reshape_copy": "aten.copy.out",
        "rms_norm_scratch_copy": "aten.rms_norm.default+aten.copy.out",
        "sdpa_math_scratch_copy": "aten.scaled_dot_product_attention.default+aten.copy.out",
    }
)

_ALIAS_TARGETS = frozenset(
    {
        "aten.view.default",
        "aten.transpose.int",
        "aten.slice.Tensor",
        "aten.unsqueeze.default",
        "aten.permute.default",
        "aten.select.int",
        "aten.squeeze.dim",
        "aten.detach.default",
    }
)
_CONDITIONAL_ALIAS_TARGETS = frozenset({"aten.reshape.default", "operator.getitem"})

_FLOAT_DTYPES = frozenset({"torch.float16", "torch.bfloat16", "torch.float32"})
_INDEX_DTYPES = frozenset({"torch.int32", "torch.int64"})


def supported_operator_targets() -> Tuple[str, ...]:
    """Return the complete version-one front-end allowlist."""

    return tuple(
        sorted(
            tuple(_COMPUTE_ADAPTERS)
            + tuple(_ALIAS_TARGETS)
            + tuple(_CONDITIONAL_ALIAS_TARGETS)
        )
    )


def operator_allowlist_hash() -> str:
    payload = {
        "version": OPERATOR_ALLOWLIST_VERSION,
        "compute": dict(sorted(_COMPUTE_ADAPTERS.items())),
        "backend_targets": dict(sorted(_BACKEND_TARGETS.items())),
        "aliases": sorted(_ALIAS_TARGETS),
        "conditional_aliases": {
            "aten.reshape.default": "alias_or_planned_copy"
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def backend_operator_targets() -> Mapping[str, str]:
    """Map planner adapter names to exact ATen entrypoints used by a backend."""

    return _BACKEND_TARGETS


def capture_inference(
    module: Any,
    example_args: Sequence[Any],
    example_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    state_provider: Any = None,
    torch_module: Any = None,
) -> CapturedInference:
    """Strictly export ``module`` and build a managed inference plan.

    Dynamic-shape declarations and non-strict export are intentionally absent
    from this API.  A training-mode module is rejected rather than mutated by
    calling ``eval()`` on the caller's behalf.
    """

    if getattr(module, "training", False):
        raise PlanError("inference capture requires module.eval()")
    if isinstance(example_args, (str, bytes)) or not isinstance(example_args, Sequence):
        raise TypeError("example_args must be a sequence")
    kwargs = dict(example_kwargs or {})
    torch_value = torch_module
    if torch_value is None:
        try:
            torch_value = importlib.import_module("torch")
        except (ImportError, OSError) as error:
            raise PlanError("PyTorch is required for torch.export capture") from error
    export_namespace = getattr(torch_value, "export", None)
    export_function = getattr(export_namespace, "export", None)
    if export_function is None or not callable(export_function):
        raise PlanError("this PyTorch build does not expose torch.export.export")
    exported = export_function(module, tuple(example_args), kwargs, strict=True)
    return CapturedInference(
        exported_program=exported,
        plan=build_inference_plan(exported, state_provider=state_provider),
    )


def build_inference_plan(exported_program: Any, *, state_provider: Any = None) -> InferencePlan:
    """Validate and normalize one static ``ExportedProgram``.

    The returned plan retains no FX node objects or raw CUDA/host addresses.
    Process-local storage identities are reduced to deterministic tied-storage
    groups before hashing.
    """

    graph_module = getattr(exported_program, "graph_module", None)
    graph = getattr(graph_module, "graph", None)
    raw_nodes = list(getattr(graph, "nodes", ()) or ())
    if not raw_nodes:
        raise PlanError("exported program has no FX graph")

    input_specs, output_specs = _graph_signature_specs(exported_program)
    input_by_name = {item[0]: item for item in input_specs}
    state_source = _StateSource(exported_program, state_provider)

    names: Dict[int, str] = {}
    node_by_name: Dict[str, Any] = {}
    node_index: Dict[str, int] = {}
    for index, node in enumerate(raw_nodes):
        name = getattr(node, "name", None)
        if not isinstance(name, str) or not name or name in node_by_name:
            raise PlanError("FX node names must be unique non-empty strings")
        names[id(node)] = name
        node_by_name[name] = node
        node_index[name] = index

    output_names = tuple(item[0] for item in output_specs)
    _validate_output_node(raw_nodes, output_names, names)

    uses: Dict[str, List[int]] = {name: [] for name in node_by_name}
    for index, node in enumerate(raw_nodes):
        for source in _input_nodes(node):
            source_name = names.get(id(source))
            if source_name is None or node_index[source_name] >= index:
                raise PlanError(
                    "FX graph is not in topological order",
                    node=getattr(node, "name", ""),
                )
            uses[source_name].append(index)

    values: Dict[str, ManagedValue] = {}
    state_targets: Dict[str, str] = {}
    state_groups: Dict[Hashable, Tuple[str, int]] = {}
    state_group_index = 0
    input_names: List[str] = []

    for index, node in enumerate(raw_nodes):
        name = names[id(node)]
        op = str(getattr(node, "op", ""))
        if op == "output":
            continue
        spec = _tensor_spec(node, name)
        first_use = uses[name][0] if uses[name] else index
        last_use = uses[name][-1] if uses[name] else index
        escapes = name in output_names
        if op == "placeholder":
            binding = input_by_name.get(name)
            if binding is None:
                raise PlanError("placeholder is absent from graph signature", node=name)
            _arg_name, kind_name, target = binding
            if kind_name == "USER_INPUT":
                storage_id = "input:{}".format(len(input_names))
                input_names.append(name)
                values[name] = ManagedValue(
                    name=name,
                    kind=ValueKind.INPUT,
                    spec=spec,
                    producer_index=index,
                    first_use_index=first_use,
                    last_use_index=last_use,
                    storage_id=storage_id,
                    storage_bytes=spec.span_bytes,
                    escapes_graph=escapes,
                )
                continue
            kind = _state_kind(kind_name, name)
            if not target:
                raise PlanError("state placeholder has no target", node=name)
            info = state_source.describe(target)
            _validate_state_spec(name, spec, info)
            group = state_groups.get(info.storage_key)
            if group is None:
                storage_id = "state:{}".format(state_group_index)
                state_group_index += 1
                state_groups[info.storage_key] = (storage_id, info.storage_bytes)
                storage_bytes = info.storage_bytes
            else:
                storage_id, storage_bytes = group
                if storage_bytes != info.storage_bytes:
                    raise PlanError("tied state reports inconsistent storage sizes", node=name)
            if info.offset_bytes + spec.span_bytes > storage_bytes:
                raise PlanError("state tensor exceeds its backing storage", node=name)
            state_targets[name] = target
            values[name] = ManagedValue(
                name=name,
                kind=kind,
                spec=spec,
                producer_index=index,
                first_use_index=first_use,
                last_use_index=last_use,
                storage_id=storage_id,
                storage_bytes=storage_bytes,
                storage_offset_bytes=info.offset_bytes,
                state_target=target,
                escapes_graph=escapes,
            )
            continue
        if op != "call_function":
            raise PlanError("only functional ATen nodes are supported", node=name, target=op)
        target = _target_name(getattr(node, "target", None))
        if (
            target not in _COMPUTE_ADAPTERS
            and target not in _ALIAS_TARGETS
            and target not in _CONDITIONAL_ALIAS_TARGETS
        ):
            raise PlanError(
                "operator is not in the managed inference allowlist",
                node=name,
                target=target,
            )
        node_is_alias = _node_is_alias(node, target)
        if target == "operator.getitem" and not node_is_alias:
            raise PlanError(
                "getitem is supported only when metadata proves a tensor alias",
                node=name,
                target=target,
            )
        if node_is_alias:
            inputs = _input_nodes(node)
            if not inputs:
                raise PlanError("alias operator has no tensor input", node=name, target=target)
            source_name = names[id(inputs[0])]
            source = values[source_name]
            root = values[source.alias_of] if source.alias_of is not None else source
            if spec.dtype != source.spec.dtype or spec.element_size != source.spec.element_size:
                raise PlanError("alias changes dtype", node=name, target=target)
            offset_bytes = spec.storage_offset_elements * spec.element_size
            if offset_bytes + spec.span_bytes > root.storage_bytes:
                raise PlanError("alias exceeds base storage", node=name, target=target)
            values[name] = ManagedValue(
                name=name,
                kind=source.kind,
                spec=spec,
                producer_index=index,
                first_use_index=first_use,
                last_use_index=last_use,
                storage_id=root.storage_id,
                storage_bytes=root.storage_bytes,
                storage_offset_bytes=offset_bytes,
                alias_of=root.name,
                state_target=root.state_target,
                escapes_graph=escapes,
            )
        else:
            values[name] = ManagedValue(
                name=name,
                kind=ValueKind.ACTIVATION,
                spec=spec,
                producer_index=index,
                first_use_index=first_use,
                last_use_index=last_use,
                storage_id="pending:{}".format(name),
                storage_bytes=spec.span_bytes,
                escapes_graph=escapes,
            )

    values = _propagate_alias_liveness(values)
    _validate_graph_semantics(raw_nodes, names, values)
    values, slots = _color_activation_slots(values)
    planned_nodes = _plan_nodes(raw_nodes, names, values)
    logical_state_bytes = sum(size for _storage_id, size in state_groups.values())
    logical_activation_bytes = sum(slot.capacity_bytes for slot in slots)
    peak_live = _peak_live_activation_bytes(values)

    provisional = {
        "planner_version": PLANNER_VERSION,
        "allowlist_hash": operator_allowlist_hash(),
        "inputs": input_names,
        "outputs": output_names,
        "state_targets": state_targets,
        "values": [_canonical_value(values[name]) for name in sorted(values)],
        "nodes": [_canonical_node(node) for node in planned_nodes],
        "slots": [
            {"index": slot.index, "bytes": slot.capacity_bytes, "values": slot.values}
            for slot in slots
        ],
    }
    graph_hash = hashlib.sha256(_canonical_json(provisional).encode("utf-8")).hexdigest()
    return InferencePlan(
        version=PLANNER_VERSION,
        graph_hash=graph_hash,
        allowlist_hash=operator_allowlist_hash(),
        values=MappingProxyType(dict(values)),
        nodes=tuple(planned_nodes),
        input_names=tuple(input_names),
        output_names=output_names,
        state_targets=MappingProxyType(dict(state_targets)),
        activation_slots=tuple(slots),
        logical_state_bytes=logical_state_bytes,
        logical_activation_bytes=logical_activation_bytes,
        peak_live_activation_bytes=peak_live,
    )


def embedding_row_ranges(
    indices: Iterable[Any],
    *,
    vocabulary_rows: int,
    row_bytes: int,
    storage_id: str,
    storage_offset_bytes: int = 0,
    value_name: str = "embedding_weight",
) -> Tuple[ManagedRange, ...]:
    """Convert host-visible embedding indices into merged row byte ranges."""

    if vocabulary_rows <= 0 or row_bytes <= 0 or storage_offset_bytes < 0:
        raise ValueError("invalid embedding storage geometry")
    rows = set()
    for raw in indices:
        if isinstance(raw, bool):
            raise ValueError("embedding indices must be integers")
        try:
            row = int(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("embedding indices must be integers") from error
        if row < 0 or row >= vocabulary_rows:
            raise IndexError("embedding index {} is out of range".format(row))
        rows.add(row)
    if not rows:
        raise ValueError("embedding indices cannot be empty")
    ordered = sorted(rows)
    ranges: List[ManagedRange] = []
    begin = previous = ordered[0]
    for row in ordered[1:]:
        if row == previous + 1:
            previous = row
            continue
        ranges.append(
            ManagedRange(
                storage_id=storage_id,
                offset_bytes=storage_offset_bytes + begin * row_bytes,
                length_bytes=(previous - begin + 1) * row_bytes,
                mode=AccessMode.READ,
                values=(value_name,),
            )
        )
        begin = previous = row
    ranges.append(
        ManagedRange(
            storage_id=storage_id,
            offset_bytes=storage_offset_bytes + begin * row_bytes,
            length_bytes=(previous - begin + 1) * row_bytes,
            mode=AccessMode.READ,
            values=(value_name,),
        )
    )
    return tuple(ranges)


def merge_managed_ranges(ranges: Iterable[ManagedRange]) -> Tuple[ManagedRange, ...]:
    """Merge overlapping/adjacent ranges without weakening access modes."""

    grouped: Dict[str, List[ManagedRange]] = {}
    for item in ranges:
        if item.offset_bytes < 0 or item.length_bytes <= 0:
            raise ValueError("managed ranges must be non-empty and non-negative")
        grouped.setdefault(item.storage_id, []).append(item)
    result: List[ManagedRange] = []
    for storage_id in sorted(grouped):
        items = sorted(
            grouped[storage_id],
            key=lambda value: (value.offset_bytes, value.length_bytes),
        )
        current = items[0]
        for item in items[1:]:
            current_end = current.offset_bytes + current.length_bytes
            item_end = item.offset_bytes + item.length_bytes
            if item.offset_bytes <= current_end:
                mode = _merge_modes(current.mode, item.mode)
                current = ManagedRange(
                    storage_id=storage_id,
                    offset_bytes=current.offset_bytes,
                    length_bytes=max(current_end, item_end) - current.offset_bytes,
                    mode=mode,
                    values=tuple(sorted(set(current.values + item.values))),
                )
            else:
                result.append(current)
                current = item
        result.append(current)
    return tuple(result)


class _StateSource:
    def __init__(self, exported_program: Any, provider: Any) -> None:
        self._provider = provider
        self._state = dict(getattr(exported_program, "state_dict", {}) or {})
        self._constants = dict(getattr(exported_program, "constants", {}) or {})

    def describe(self, target: str) -> StateTensorInfo:
        if self._provider is not None:
            describe = getattr(self._provider, "describe", None)
            if describe is None or not callable(describe):
                raise PlanError("state provider does not implement describe()")
            info = describe(target)
            if not isinstance(info, StateTensorInfo):
                raise PlanError("state provider returned an invalid descriptor")
            return info
        tensor = self._state.get(target, self._constants.get(target))
        if tensor is None:
            raise PlanError("state target {!r} is unavailable".format(target))
        return describe_cpu_state_tensor(target, tensor)


def describe_cpu_state_tensor(target: str, tensor: Any) -> StateTensorInfo:
    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if str(device_type) != "cpu":
        raise PlanError("state tensors must reside in pageable CPU memory")
    spec = _tensor_spec_from_value(tensor, target)
    storage_method = getattr(tensor, "untyped_storage", None)
    if storage_method is None or not callable(storage_method):
        raise PlanError("state tensor does not expose untyped_storage()")
    storage = storage_method()
    nbytes_value = getattr(storage, "nbytes", None)
    storage_bytes = int(nbytes_value() if callable(nbytes_value) else nbytes_value)
    data_ptr = int(storage.data_ptr())
    offset_bytes = spec.storage_offset_elements * spec.element_size
    if storage_bytes <= 0 or data_ptr == 0:
        raise PlanError("empty state storages are unsupported")
    return StateTensorInfo(
        target=target,
        tensor=tensor,
        storage_key=(data_ptr, storage_bytes),
        storage_bytes=storage_bytes,
        offset_bytes=offset_bytes,
        spec=spec,
    )


def _graph_signature_specs(
    exported_program: Any,
) -> Tuple[List[Tuple[str, str, Optional[str]]], List[Tuple[str, str]]]:
    signature = getattr(exported_program, "graph_signature", None)
    if signature is None:
        raise PlanError("exported program has no graph signature")
    inputs: List[Tuple[str, str, Optional[str]]] = []
    for spec in getattr(signature, "input_specs", ()):
        arg = getattr(spec, "arg", None)
        name = getattr(arg, "name", None)
        if not isinstance(name, str) or not name:
            raise PlanError("only tensor graph inputs are supported")
        kind = _enum_name(getattr(spec, "kind", ""))
        if kind not in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR", "USER_INPUT"}:
            raise PlanError("unsupported graph input kind {!r}".format(kind))
        target = getattr(spec, "target", None)
        inputs.append((name, kind, str(target) if target is not None else None))
    outputs: List[Tuple[str, str]] = []
    for spec in getattr(signature, "output_specs", ()):
        kind = _enum_name(getattr(spec, "kind", ""))
        if kind != "USER_OUTPUT":
            raise PlanError("mutation and non-user graph outputs are unsupported")
        name = getattr(getattr(spec, "arg", None), "name", None)
        if not isinstance(name, str) or not name:
            raise PlanError("only tensor graph outputs are supported")
        outputs.append((name, kind))
    if not outputs:
        raise PlanError("at least one tensor output is required")
    return inputs, outputs


def _validate_output_node(
    nodes: Sequence[Any], outputs: Tuple[str, ...], names: Mapping[int, str]
) -> None:
    output_nodes = [node for node in nodes if str(getattr(node, "op", "")) == "output"]
    if len(output_nodes) != 1 or output_nodes[0] is not nodes[-1]:
        raise PlanError("graph must end in exactly one output node")
    actual = tuple(names[id(node)] for node in _input_nodes(output_nodes[0]))
    if actual != outputs:
        raise PlanError("graph signature outputs do not match the FX output node")


def _state_kind(kind: str, node: str) -> ValueKind:
    mapping = {
        "PARAMETER": ValueKind.PARAMETER,
        "BUFFER": ValueKind.BUFFER,
        "CONSTANT_TENSOR": ValueKind.CONSTANT,
    }
    try:
        return mapping[kind]
    except KeyError as error:
        raise PlanError("unsupported state input kind", node=node) from error


def _validate_state_spec(name: str, graph: TensorSpec, state: StateTensorInfo) -> None:
    actual = state.spec
    if (graph.dtype, graph.element_size, graph.sizes, graph.strides) != (
        actual.dtype,
        actual.element_size,
        actual.sizes,
        actual.strides,
    ):
        raise PlanError("state tensor metadata differs from the exported graph", node=name)


def _tensor_spec(node: Any, name: str) -> TensorSpec:
    meta = getattr(node, "meta", {}) or {}
    if "val" not in meta:
        raise PlanError("tensor metadata is missing", node=name)
    return _tensor_spec_from_value(meta["val"], name)


def _tensor_spec_from_value(value: Any, name: str) -> TensorSpec:
    size_method = getattr(value, "size", None)
    stride_method = getattr(value, "stride", None)
    element_method = getattr(value, "element_size", None)
    if not callable(size_method) or not callable(stride_method) or not callable(element_method):
        raise PlanError("node does not produce exactly one tensor", node=name)
    sizes = tuple(_static_int(item, "shape", name) for item in size_method())
    strides = tuple(_static_int(item, "stride", name) for item in stride_method())
    if len(sizes) != len(strides) or not sizes or any(item <= 0 for item in sizes):
        raise PlanError("zero-sized, scalar, or malformed tensors are unsupported", node=name)
    if any(item < 0 for item in strides):
        raise PlanError("negative strides are unsupported", node=name)
    element_size = _static_int(element_method(), "element size", name)
    if element_size <= 0:
        raise PlanError("invalid tensor element size", node=name)
    storage_offset_method = getattr(value, "storage_offset", None)
    storage_offset = _static_int(
        storage_offset_method() if callable(storage_offset_method) else 0,
        "storage offset",
        name,
    )
    if storage_offset < 0:
        raise PlanError("negative storage offsets are unsupported", node=name)
    numel = 1
    maximum = 0
    for size, stride in zip(sizes, strides):
        numel = _checked_mul(numel, size, name)
        maximum = _checked_add(maximum, _checked_mul(size - 1, stride, name), name)
    span_bytes = _checked_mul(maximum + 1, element_size, name)
    dtype = str(getattr(value, "dtype", ""))
    if not dtype:
        raise PlanError("tensor dtype is unavailable", node=name)
    return TensorSpec(
        dtype=dtype,
        element_size=element_size,
        sizes=sizes,
        strides=strides,
        storage_offset_elements=storage_offset,
        span_bytes=span_bytes,
        numel=numel,
    )


def _static_int(value: Any, field: str, node: str) -> int:
    if isinstance(value, bool):
        raise PlanError("{} must be a static integer".format(field), node=node)
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PlanError("{} must be a static integer".format(field), node=node) from error
    if result < -(1 << 63) or result > (1 << 63) - 1:
        raise PlanError("{} exceeds the signed 64-bit range".format(field), node=node)
    return result


def _checked_mul(left: int, right: int, node: str) -> int:
    result = left * right
    if result > (1 << 63) - 1:
        raise PlanError("tensor geometry overflows signed 64-bit", node=node)
    return result


def _checked_add(left: int, right: int, node: str) -> int:
    result = left + right
    if result > (1 << 63) - 1:
        raise PlanError("tensor geometry overflows signed 64-bit", node=node)
    return result


def _input_nodes(node: Any) -> Tuple[Any, ...]:
    values = getattr(node, "all_input_nodes", ())
    if callable(values):
        values = values()
    return tuple(values or ())


def _target_name(target: Any) -> str:
    if target is operator.getitem:
        return "operator.getitem"
    value = str(target)
    return value if value else "<unknown>"


def _node_is_alias(node: Any, target: str) -> bool:
    if target in _ALIAS_TARGETS:
        return True
    if target not in _CONDITIONAL_ALIAS_TARGETS:
        return False
    inputs = _input_nodes(node)
    if not inputs:
        return False
    source_value = (getattr(inputs[0], "meta", {}) or {}).get("val")
    output_value = (getattr(node, "meta", {}) or {}).get("val")
    source_key = _tensor_storage_identity(source_value)
    output_key = _tensor_storage_identity(output_value)
    return source_key is not None and source_key == output_key


def _tensor_storage_identity(value: Any) -> Optional[Tuple[str, int]]:
    storage_method = getattr(value, "untyped_storage", None)
    if storage_method is None or not callable(storage_method):
        return None
    try:
        storage = storage_method()
        cdata = getattr(storage, "_cdata", None)
        if cdata is not None:
            return ("cdata", int(cdata))
        data_ptr = getattr(storage, "data_ptr", None)
        if callable(data_ptr):
            return ("data_ptr", int(data_ptr()))
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None
    return None


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).rsplit(".", 1)[-1].upper()


def _validate_graph_semantics(
    nodes: Sequence[Any],
    names: Mapping[int, str],
    values: Mapping[str, ManagedValue],
) -> None:
    for node in nodes:
        name = names[id(node)]
        if str(getattr(node, "op", "")) != "call_function":
            continue
        target = _target_name(getattr(node, "target", None))
        inputs = tuple(values[names[id(item)]] for item in _input_nodes(node))
        output = values[name]
        if output.is_alias:
            continue
        if output.spec.dtype not in _FLOAT_DTYPES:
            raise PlanError("compute output dtype is unsupported", node=name, target=target)
        if target == "aten.embedding.default":
            _validate_embedding(node, name, inputs, output)
        elif target == "aten.linear.default":
            _require_float_inputs(name, target, inputs, minimum=2)
            if len(inputs[1].spec.sizes) != 2:
                raise PlanError("linear weight must be rank two", node=name, target=target)
        elif target == "aten.rms_norm.default":
            _require_float_inputs(name, target, inputs, minimum=1)
        elif target == "aten.scaled_dot_product_attention.default":
            _require_float_inputs(name, target, inputs, minimum=3)
            _validate_sdpa_constants(node, name)
        elif target == "aten.mm.default":
            _require_float_inputs(name, target, inputs, minimum=2)
            if len(inputs[0].spec.sizes) != 2 or len(inputs[1].spec.sizes) != 2:
                raise PlanError("mm inputs must be rank two", node=name, target=target)
        elif target == "aten.addmm.default":
            _require_float_inputs(name, target, inputs, minimum=3)
            if len(inputs[1].spec.sizes) != 2 or len(inputs[2].spec.sizes) != 2:
                raise PlanError("addmm matrix inputs must be rank two", node=name, target=target)
        elif target == "aten.bmm.default":
            _require_float_inputs(name, target, inputs, minimum=2)
            if len(inputs[0].spec.sizes) != 3 or len(inputs[1].spec.sizes) != 3:
                raise PlanError("bmm inputs must be rank three", node=name, target=target)
        else:
            _require_float_inputs(name, target, inputs, minimum=1)


def _require_float_inputs(
    name: str, target: str, inputs: Sequence[ManagedValue], minimum: int
) -> None:
    if len(inputs) < minimum or any(item.spec.dtype not in _FLOAT_DTYPES for item in inputs):
        raise PlanError("operator has unsupported tensor inputs", node=name, target=target)


def _validate_embedding(
    node: Any,
    name: str,
    inputs: Sequence[ManagedValue],
    output: ManagedValue,
) -> None:
    if len(inputs) != 2:
        raise PlanError("embedding requires weight and indices tensors", node=name)
    weight, indices = inputs
    if weight.spec.dtype not in _FLOAT_DTYPES or indices.spec.dtype not in _INDEX_DTYPES:
        raise PlanError("embedding dtype combination is unsupported", node=name)
    if (
        len(weight.spec.sizes) != 2
        or weight.spec.strides[1] != 1
        or weight.spec.strides[0] != weight.spec.sizes[1]
    ):
        raise PlanError("embedding weight must be dense row-major", node=name)
    expected = indices.spec.sizes + (weight.spec.sizes[1],)
    if output.spec.sizes != expected:
        raise PlanError("embedding output shape is inconsistent", node=name)
    args = tuple(getattr(node, "args", ()) or ())
    scale = args[3] if len(args) > 3 else False
    sparse = args[4] if len(args) > 4 else False
    if bool(scale) or bool(sparse):
        raise PlanError("embedding scale_grad_by_freq and sparse modes are unsupported", node=name)
    if indices.kind is not ValueKind.INPUT:
        raise PlanError("embedding indices must be a host-visible user input", node=name)


def _validate_sdpa_constants(node: Any, name: str) -> None:
    args = tuple(getattr(node, "args", ()) or ())
    kwargs = dict(getattr(node, "kwargs", {}) or {})
    mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
    dropout = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
    causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
    enable_gqa = kwargs.get("enable_gqa", False)
    if mask is not None or float(dropout) != 0.0 or not bool(causal) or bool(enable_gqa):
        raise PlanError(
            "SDPA v1 requires attn_mask=None, dropout_p=0, is_causal=True, enable_gqa=False",
            node=name,
        )


def _color_activation_slots(
    values: Mapping[str, ManagedValue],
) -> Tuple[Dict[str, ManagedValue], List[ActivationSlot]]:
    material = sorted(
        (
            value
            for value in values.values()
            if value.kind is ValueKind.ACTIVATION and not value.is_alias
        ),
        key=lambda value: (value.producer_index, value.name),
    )
    mutable = dict(values)
    slots: List[Dict[str, Any]] = []
    for value in material:
        available = [slot for slot in slots if slot["end"] < value.producer_index]
        fitting = [slot for slot in available if slot["capacity"] >= value.spec.span_bytes]
        if fitting:
            slot = min(fitting, key=lambda item: (item["capacity"], item["index"]))
        elif available:
            slot = min(available, key=lambda item: (item["capacity"], item["index"]))
        else:
            slot = {"index": len(slots), "capacity": 0, "end": -1, "values": []}
            slots.append(slot)
        slot["capacity"] = max(slot["capacity"], value.spec.span_bytes)
        slot["end"] = value.last_use_index
        slot["values"].append(value.name)
        storage_id = "activation:{}".format(slot["index"])
        mutable[value.name] = replace(
            value,
            storage_id=storage_id,
            storage_bytes=slot["capacity"],
            activation_slot=slot["index"],
        )
    # Slot capacities can grow after earlier values were assigned.
    for name, value in list(mutable.items()):
        if value.activation_slot is not None:
            slot = slots[value.activation_slot]
            mutable[name] = replace(value, storage_bytes=slot["capacity"])
    # Aliases inherit the final slot and capacity from their root.
    for name, value in list(mutable.items()):
        if value.alias_of is not None:
            root = mutable[value.alias_of]
            mutable[name] = replace(
                value,
                storage_id=root.storage_id,
                storage_bytes=root.storage_bytes,
                activation_slot=root.activation_slot,
            )
    output = [
        ActivationSlot(
            index=slot["index"],
            storage_id="activation:{}".format(slot["index"]),
            capacity_bytes=slot["capacity"],
            values=tuple(slot["values"]),
        )
        for slot in slots
    ]
    return mutable, output


def _propagate_alias_liveness(values: Mapping[str, ManagedValue]) -> Dict[str, ManagedValue]:
    """Keep a base storage live until every derived alias is dead."""

    mutable = dict(values)
    for value in values.values():
        if value.alias_of is None:
            continue
        root = mutable[value.alias_of]
        mutable[root.name] = replace(
            root,
            last_use_index=max(root.last_use_index, value.last_use_index),
            escapes_graph=root.escapes_graph or value.escapes_graph,
        )
    return mutable


def _plan_nodes(
    nodes: Sequence[Any],
    names: Mapping[int, str],
    values: Mapping[str, ManagedValue],
) -> List[PlannedNode]:
    compute_indices = [
        index
        for index, node in enumerate(nodes)
        if str(getattr(node, "op", "")) == "call_function"
    ]
    launch_indices = [
        index for index in compute_indices if not values[names[id(nodes[index])]].is_alias
    ]
    launch_position = {index: position for position, index in enumerate(launch_indices)}
    result: List[PlannedNode] = []
    for index in compute_indices:
        node = nodes[index]
        name = names[id(node)]
        target = _target_name(getattr(node, "target", None))
        kind = NodeKind.ALIAS if values[name].is_alias else NodeKind.COMPUTE
        inputs = tuple(names[id(item)] for item in _input_nodes(node))
        accesses: List[ManagedRange] = []
        embedding = None
        if kind is NodeKind.COMPUTE:
            for input_name in inputs:
                value = values[input_name]
                if target == "aten.embedding.default" and input_name == inputs[0]:
                    continue
                accesses.append(_range_for_value(value, AccessMode.READ))
            accesses.append(_range_for_value(values[name], AccessMode.WRITE_ONLY))
            if target == "aten.embedding.default":
                weight = values[inputs[0]]
                indices = values[inputs[1]]
                embedding = EmbeddingRange(
                    weight_value=weight.name,
                    indices_value=indices.name,
                    vocabulary_rows=weight.spec.sizes[0],
                    row_bytes=weight.spec.sizes[1] * weight.spec.element_size,
                    weight_storage_id=weight.storage_id,
                    weight_storage_offset_bytes=weight.storage_offset_bytes,
                )
        release_values = tuple(
            sorted(
                value.name
                for value in values.values()
                if value.kind is ValueKind.ACTIVATION
                and not value.escapes_graph
                and value.last_use_index == index
            )
        )
        future_reads: Dict[str, int] = {}
        position = launch_position.get(index)
        future_launches = (
            launch_indices[position + 1 : position + 3] if position is not None else ()
        )
        for future in future_launches:
            for source in _input_nodes(nodes[future]):
                source_name = names[id(source)]
                candidate = values[source_name]
                if candidate.is_persistent and source_name not in inputs:
                    future_reads[source_name] = min(future, future_reads.get(source_name, future))
        prefetch_values = tuple(
            name
            for name, _future in sorted(
                future_reads.items(), key=lambda item: (item[1], item[0])
            )
        )
        adapter = (
            "alias"
            if kind is NodeKind.ALIAS
            else (
                "reshape_copy"
                if target == "aten.reshape.default"
                else _COMPUTE_ADAPTERS[target]
            )
        )
        result.append(
            PlannedNode(
                index=index,
                name=name,
                kind=kind,
                target=target,
                adapter=adapter,
                backend_target="metadata"
                if kind is NodeKind.ALIAS
                else _BACKEND_TARGETS[adapter],
                args=_normalize_argument(getattr(node, "args", ()), names),
                kwargs=MappingProxyType(
                    dict(_normalize_argument(getattr(node, "kwargs", {}), names))
                ),
                input_values=inputs,
                output_value=name,
                accesses=merge_managed_ranges(accesses) if accesses else (),
                embedding_range=embedding,
                release_values=release_values,
                prefetch_values=prefetch_values,
            )
        )
    return result


def _range_for_value(value: ManagedValue, mode: AccessMode) -> ManagedRange:
    return ManagedRange(
        storage_id=value.storage_id,
        offset_bytes=value.storage_offset_bytes,
        length_bytes=value.spec.span_bytes,
        mode=mode,
        values=(value.name,),
    )


def _peak_live_activation_bytes(values: Mapping[str, ManagedValue]) -> int:
    material = [
        value
        for value in values.values()
        if value.kind is ValueKind.ACTIVATION and not value.is_alias
    ]
    if not material:
        return 0
    maximum_index = max(value.last_use_index for value in material)
    return max(
        sum(
            value.spec.span_bytes
            for value in material
            if value.producer_index <= index <= value.last_use_index
        )
        for index in range(maximum_index + 1)
    )


def _normalize_argument(value: Any, names: Mapping[int, str]) -> Any:
    node_name = names.get(id(value))
    if node_name is not None:
        return ValueReference(node_name)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PlanError("non-finite graph constants are unsupported")
        return value
    if isinstance(value, tuple):
        return tuple(_normalize_argument(item, names) for item in value)
    if isinstance(value, list):
        return [_normalize_argument(item, names) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_argument(item, names)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    text = str(value)
    if text.startswith("torch."):
        return text
    raise PlanError("unsupported graph constant {!r}".format(value))


def _merge_modes(left: AccessMode, right: AccessMode) -> AccessMode:
    if left is right:
        return left
    return AccessMode.READ_WRITE


def _canonical_value(value: ManagedValue) -> Mapping[str, Any]:
    return {
        "name": value.name,
        "kind": value.kind.value,
        "dtype": value.spec.dtype,
        "element_size": value.spec.element_size,
        "sizes": value.spec.sizes,
        "strides": value.spec.strides,
        "storage_offset_elements": value.spec.storage_offset_elements,
        "span_bytes": value.spec.span_bytes,
        "producer": value.producer_index,
        "last_use": value.last_use_index,
        "storage_id": value.storage_id,
        "storage_bytes": value.storage_bytes,
        "storage_offset_bytes": value.storage_offset_bytes,
        "alias_of": value.alias_of,
        "state_target": value.state_target,
        "activation_slot": value.activation_slot,
        "escapes": value.escapes_graph,
    }


def _canonical_node(node: PlannedNode) -> Mapping[str, Any]:
    return {
        "index": node.index,
        "name": node.name,
        "kind": node.kind.value,
        "target": node.target,
        "adapter": node.adapter,
        "backend_target": node.backend_target,
        "args": _canonical_argument(node.args),
        "kwargs": _canonical_argument(node.kwargs),
        "inputs": node.input_values,
        "output": node.output_value,
        "accesses": [
            {
                "storage": item.storage_id,
                "offset": item.offset_bytes,
                "length": item.length_bytes,
                "mode": item.mode.value,
                "values": item.values,
            }
            for item in node.accesses
        ],
        "embedding": None
        if node.embedding_range is None
        else {
            "weight": node.embedding_range.weight_value,
            "indices": node.embedding_range.indices_value,
            "rows": node.embedding_range.vocabulary_rows,
            "row_bytes": node.embedding_range.row_bytes,
        },
        "release": node.release_values,
        "prefetch": node.prefetch_values,
    }


def _canonical_argument(value: Any) -> Any:
    if isinstance(value, ValueReference):
        return {"value": value.name}
    if isinstance(value, tuple):
        return {"tuple": [_canonical_argument(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_canonical_argument(item) for item in value]}
    if isinstance(value, Mapping):
        return {str(key): _canonical_argument(item) for key, item in sorted(value.items())}
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = [
    "AccessMode",
    "ActivationSlot",
    "CapturedInference",
    "EmbeddingRange",
    "InferencePlan",
    "ManagedRange",
    "ManagedValue",
    "NodeKind",
    "OPERATOR_ALLOWLIST_VERSION",
    "PLANNER_VERSION",
    "PlanError",
    "PlannedNode",
    "StateTensorInfo",
    "TensorSpec",
    "ValueKind",
    "ValueReference",
    "backend_operator_targets",
    "build_inference_plan",
    "capture_inference",
    "describe_cpu_state_tensor",
    "embedding_row_ranges",
    "merge_managed_ranges",
    "operator_allowlist_hash",
    "supported_operator_targets",
]
