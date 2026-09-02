"""Conservative tensor-lifetime metadata for the xVRAM PyTorch adapter.

The helpers in this module only classify objects.  They never inspect a CUDA
address, move storage, or make an object eligible for eviction.  A runtime must
still establish stream/event safety before it acts on any hint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class TensorLifetime(str, Enum):
    """Coarse lifetime classes understood by the Phase 4 planner."""

    UNKNOWN = "unknown"
    EXTERNAL = "external"
    PERSISTENT = "persistent"
    ITERATION = "iteration"
    EPHEMERAL = "ephemeral"


class TensorRole(str, Enum):
    """Roles callers can supply without relying on tensor heuristics."""

    INPUT = "input"
    OUTPUT = "output"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    OPTIMIZER_STATE = "optimizer_state"
    ACTIVATION = "activation"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TensorLifetimeHint:
    """An advisory classification, never an eviction authorization."""

    lifetime: TensorLifetime
    role: TensorRole
    reason: str
    explicit: bool
    advisory_only: bool = True


_ROLE_LIFETIMES: Mapping[TensorRole, TensorLifetime] = {
    TensorRole.INPUT: TensorLifetime.EXTERNAL,
    TensorRole.OUTPUT: TensorLifetime.EXTERNAL,
    TensorRole.PARAMETER: TensorLifetime.PERSISTENT,
    TensorRole.BUFFER: TensorLifetime.PERSISTENT,
    TensorRole.OPTIMIZER_STATE: TensorLifetime.PERSISTENT,
    TensorRole.ACTIVATION: TensorLifetime.ITERATION,
    TensorRole.TEMPORARY: TensorLifetime.EPHEMERAL,
    TensorRole.UNKNOWN: TensorLifetime.UNKNOWN,
}


def _coerce_role(role: Optional[object]) -> TensorRole:
    if role is None:
        return TensorRole.UNKNOWN
    if isinstance(role, TensorRole):
        return role
    try:
        return TensorRole(str(role).strip().lower())
    except ValueError as error:
        valid = ", ".join(item.value for item in TensorRole)
        raise ValueError("unknown tensor role {!r}; expected one of {}".format(role, valid)) from error


def classify_tensor(
    tensor: Any = None,
    *,
    role: Optional[object] = None,
) -> TensorLifetimeHint:
    """Return a conservative, metadata-only lifetime hint.

    ``tensor`` is intentionally not introspected.  Properties such as
    ``requires_grad`` or ``is_leaf`` are insufficient to distinguish model
    parameters from caller-owned inputs, and querying a device pointer would
    create additional synchronization and lifetime hazards.
    """

    del tensor
    normalized = _coerce_role(role)
    if normalized is TensorRole.UNKNOWN:
        return TensorLifetimeHint(
            lifetime=TensorLifetime.UNKNOWN,
            role=normalized,
            reason="no explicit ownership role was supplied",
            explicit=False,
        )

    return TensorLifetimeHint(
        lifetime=_ROLE_LIFETIMES[normalized],
        role=normalized,
        reason="classified from the caller-supplied role '{}'".format(normalized.value),
        explicit=True,
    )


def classify_module_tensors(module: Any) -> Dict[str, TensorLifetimeHint]:
    """Classify named module parameters and buffers as persistent.

    The function is deliberately duck-typed so metadata tests do not require a
    PyTorch installation.  It does not walk optimizer state or arbitrary
    attributes.
    """

    hints: Dict[str, TensorLifetimeHint] = {}
    for name, _tensor in _named_items(module, "named_parameters"):
        hints[name] = TensorLifetimeHint(
            lifetime=TensorLifetime.PERSISTENT,
            role=TensorRole.PARAMETER,
            reason="registered module parameter",
            explicit=True,
        )
    for name, _tensor in _named_items(module, "named_buffers"):
        hints[name] = TensorLifetimeHint(
            lifetime=TensorLifetime.PERSISTENT,
            role=TensorRole.BUFFER,
            reason="registered module buffer",
            explicit=True,
        )
    return hints


def _named_items(owner: Any, method_name: str) -> Iterable[Tuple[str, Any]]:
    method = getattr(owner, method_name, None)
    if method is None or not callable(method):
        raise TypeError("object does not provide callable {}()".format(method_name))

    try:
        values = method(recurse=True, remove_duplicate=True)
    except TypeError:
        try:
            values = method(recurse=True)
        except TypeError:
            values = method()

    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("{}() must yield (name, tensor) pairs".format(method_name))
        name, tensor = item
        if not isinstance(name, str) or not name:
            raise ValueError("registered tensor names must be non-empty strings")
        yield name, tensor
