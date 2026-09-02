"""Static FX next-use analysis for Phase 4 scheduling hints.

The planner is intentionally side-effect free.  Its release and prefetch
candidates are advisory graph facts; they do not imply that CUDA work has
retired and never trigger mapping, eviction, or transfer operations.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FxValueUse:
    """Producer and statically visible users of one FX value."""

    name: str
    producer_index: int
    producer_op: str
    use_indices: Tuple[int, ...]
    escapes_graph: bool

    def next_use_after(self, index: int) -> Optional[int]:
        """Return the first consumer strictly after ``index``."""

        position = bisect_right(self.use_indices, index)
        if position == len(self.use_indices):
            return None
        return self.use_indices[position]


@dataclass(frozen=True)
class FxReadHint:
    """Next-use information for a value read by the current operation."""

    value_name: str
    next_use_index: Optional[int]


@dataclass(frozen=True)
class FxStepPlan:
    """Advisory information applied after one graph node has submitted its work.

    Release candidates still require completion of that node's CUDA event before
    storage can be reclaimed. Prefetch candidates name values produced strictly
    before this node; a value produced by the current node is already resident.
    """

    index: int
    node_name: str
    node_op: str
    reads: Tuple[FxReadHint, ...]
    release_candidates: Tuple[str, ...]
    prefetch_candidates: Tuple[str, ...]


@dataclass(frozen=True)
class FxNextUsePlan:
    """Immutable, non-executing representation of an FX schedule."""

    node_names: Tuple[str, ...]
    values: Mapping[str, FxValueUse]
    steps: Tuple[FxStepPlan, ...]
    prefetch_distance: int
    advisory_only: bool = True

    def value(self, name: str) -> FxValueUse:
        try:
            return self.values[name]
        except KeyError as error:
            raise KeyError("unknown FX value {!r}".format(name)) from error


def analyze_next_uses(graph_or_module: Any, *, prefetch_distance: int = 2) -> FxNextUsePlan:
    """Build deterministic next-use and release-candidate metadata.

    ``graph_or_module`` may be a ``torch.fx.GraphModule``, a ``torch.fx.Graph``,
    or a duck-typed object exposing the same ``nodes``/``all_input_nodes``
    surface.  This keeps CPU-only contract tests independent of PyTorch.

    Release candidates exclude placeholders, ``get_attr`` values and anything
    returned from the graph.  Even included values require a completed CUDA
    event boundary before a runtime may reclaim their storage.
    """

    if not isinstance(prefetch_distance, int) or isinstance(prefetch_distance, bool):
        raise TypeError("prefetch_distance must be an integer")
    if prefetch_distance < 0 or prefetch_distance > 8:
        raise ValueError("prefetch_distance must be between 0 and 8")

    graph = getattr(graph_or_module, "graph", graph_or_module)
    raw_nodes = getattr(graph, "nodes", None)
    if raw_nodes is None:
        raise TypeError("object does not expose an FX graph or nodes collection")
    nodes = list(raw_nodes)

    names: List[str] = []
    index_by_identity: Dict[int, int] = {}
    node_by_name: Dict[str, Any] = {}
    for index, node in enumerate(nodes):
        name = getattr(node, "name", None)
        op = getattr(node, "op", None)
        if not isinstance(name, str) or not name:
            raise ValueError("FX nodes must have non-empty string names")
        if name in node_by_name:
            raise ValueError("duplicate FX node name {!r}".format(name))
        if not isinstance(op, str) or not op:
            raise ValueError("FX node {!r} has no operation kind".format(name))
        names.append(name)
        index_by_identity[id(node)] = index
        node_by_name[name] = node

    inputs_by_index: List[Tuple[int, ...]] = []
    users: Dict[int, List[int]] = {index: [] for index in range(len(nodes))}
    for consumer_index, node in enumerate(nodes):
        source_indices: List[int] = []
        seen = set()
        for source in _all_input_nodes(node):
            source_index = index_by_identity.get(id(source))
            if source_index is None:
                raise ValueError(
                    "FX node {!r} references a node outside the graph".format(names[consumer_index])
                )
            if source_index >= consumer_index:
                raise ValueError(
                    "FX graph is not in topological order at {!r}".format(names[consumer_index])
                )
            if source_index not in seen:
                source_indices.append(source_index)
                seen.add(source_index)
                users[source_index].append(consumer_index)
        source_indices.sort()
        inputs_by_index.append(tuple(source_indices))

    output_indices = {
        index for index, node in enumerate(nodes) if getattr(node, "op", None) == "output"
    }
    escaped = {
        source_index
        for output_index in output_indices
        for source_index in inputs_by_index[output_index]
    }

    values: Dict[str, FxValueUse] = {}
    for index, node in enumerate(nodes):
        values[names[index]] = FxValueUse(
            name=names[index],
            producer_index=index,
            producer_op=str(getattr(node, "op")),
            use_indices=tuple(users[index]),
            escapes_graph=index in escaped,
        )

    steps: List[FxStepPlan] = []
    for index, node in enumerate(nodes):
        reads: List[FxReadHint] = []
        releases: List[str] = []
        current_value = values[names[index]]
        if _is_dead_release_candidate(current_value):
            releases.append(current_value.name)
        for source_index in inputs_by_index[index]:
            value = values[names[source_index]]
            reads.append(
                FxReadHint(
                    value_name=value.name,
                    next_use_index=value.next_use_after(index),
                )
            )
            if _is_release_candidate(value, index):
                releases.append(value.name)

        prefetches = _prefetch_candidates(
            index=index,
            distance=prefetch_distance,
            inputs_by_index=inputs_by_index,
            names=names,
        )
        current_names = {item.value_name for item in reads}
        prefetches = tuple(name for name in prefetches if name not in current_names)

        steps.append(
            FxStepPlan(
                index=index,
                node_name=names[index],
                node_op=str(getattr(node, "op")),
                reads=tuple(reads),
                release_candidates=tuple(sorted(releases)),
                prefetch_candidates=prefetches,
            )
        )

    return FxNextUsePlan(
        node_names=tuple(names),
        values=MappingProxyType(values),
        steps=tuple(steps),
        prefetch_distance=prefetch_distance,
    )


def _all_input_nodes(node: Any) -> Iterable[Any]:
    values = getattr(node, "all_input_nodes", ())
    if callable(values):
        values = values()
    if values is None:
        return ()
    return values


def _is_release_candidate(value: FxValueUse, consumer_index: int) -> bool:
    if value.producer_op in ("placeholder", "get_attr", "output"):
        return False
    if value.escapes_graph or not value.use_indices:
        return False
    return value.use_indices[-1] == consumer_index


def _is_dead_release_candidate(value: FxValueUse) -> bool:
    return (
        value.producer_op not in ("placeholder", "get_attr", "output")
        and not value.escapes_graph
        and not value.use_indices
    )


def _prefetch_candidates(
    *,
    index: int,
    distance: int,
    inputs_by_index: Sequence[Tuple[int, ...]],
    names: Sequence[str],
) -> Tuple[str, ...]:
    if distance == 0:
        return ()

    candidates: Dict[str, int] = {}
    stop = min(len(inputs_by_index), index + distance + 1)
    for future_index in range(index + 1, stop):
        for source_index in inputs_by_index[future_index]:
            # A value produced by the current or an earlier node exists and can
            # be a meaningful transfer hint.  Future-produced activations cannot.
            if source_index < index:
                candidates.setdefault(names[source_index], future_index)
    return tuple(name for name, _ in sorted(candidates.items(), key=lambda item: (item[1], item[0])))
