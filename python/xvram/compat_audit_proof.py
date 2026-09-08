"""Conditional proof models. No CUDA calls and no promotion of declared facts.

These check implications of explicit, generation-scoped facts. Their callers must
separately establish that a complete collector actually observed those facts.
Host API returns, timestamps and kernel names are intentionally not proof inputs.
"""
from dataclasses import dataclass

from .compat_audit_memory import AllocationRange, chunk_rounded_union

LIMIT = 100_000
U64 = (1 << 64) - 1


def integer(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value <= U64:
        raise ValueError("proof_integer")
    return value


def identity(value):
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("proof_identity")
    return tuple(integer(x, positive=True) for x in value)


@dataclass(frozen=True)
class TensorRange:
    allocation: tuple
    allocation_bytes: int
    tensor_offset: int
    tensor_bytes: int
    access_offset: int
    access_bytes: int
    mode: str
    base_chunk_offset: int = 0


def memory_envelopes(ranges, *, chunk_bytes):
    """Check declared tensor bounds, then allocation bounds and write aliases.

    Separate allocation IDs MUST have independently proven non-aliasing storage.
    Strided source rules may supply conservative envelopes including stride holes.
    This is not a decoder of pointers, indirect indices or kernel arguments.
    """
    integer(chunk_bytes, positive=True)
    if type(ranges) not in (list, tuple) or len(ranges) > LIMIT:
        raise ValueError("proof_range_limit")
    grouped, rounded = {}, []
    for item in ranges:
        if type(item) is not TensorRange or item.mode not in ("read", "write", "read_write"):
            raise ValueError("proof_range_type")
        key = identity(item.allocation)
        for value in (item.allocation_bytes, item.tensor_offset, item.tensor_bytes,
                      item.access_offset, item.access_bytes, item.base_chunk_offset):
            integer(value)
        if not item.allocation_bytes or item.base_chunk_offset >= chunk_bytes:
            raise ValueError("proof_allocation_extent")
        if (item.tensor_offset > item.allocation_bytes or
                item.tensor_bytes > item.allocation_bytes - item.tensor_offset):
            raise ValueError("proof_tensor_outside_allocation")
        if item.access_offset > item.tensor_bytes or item.access_bytes > item.tensor_bytes - item.access_offset:
            raise ValueError("proof_access_outside_tensor")
        begin = item.tensor_offset + item.access_offset
        end = begin + item.access_bytes
        if item.access_bytes:
            grouped.setdefault(key, []).append((begin, end, item.mode != "read"))
        rounded.append(AllocationRange(*key, item.allocation_bytes, item.base_chunk_offset, begin, item.access_bytes))
    # Sweep keeps this bounded O(n log n), including many overlapping read aliases.
    for intervals in grouped.values():
        max_end = write_end = 0
        for begin, end, writes in sorted(intervals):
            if begin < write_end or (writes and begin < max_end):
                raise ValueError("proof_overlapping_write")
            max_end = max(max_end, end)
            if writes:
                write_end = max(write_end, end)
    return chunk_rounded_union(rounded, chunk_bytes=chunk_bytes)


class DeviceOrder:
    """Single-context, explicit-stream vector-clock model, not timestamp ordering.

    All inputs are declared witness facts. Default-stream implicit rules, graphs,
    PDL and cross-context waits are outside this model. An event record snapshots
    its generation; a later re-record cannot change a previously submitted wait.
    """
    def __init__(self):
        self.streams = {}
        self.stream_generations = {}
        self.events = {}
        self.event_generations = {}
        self.records = {}
        self.nodes = {}
        self.completed = {}
        self.clock_entries = 0
        self.sealed = False
        self.poisoned = False

    def _admit(self, *, submission=True):
        if self.sealed and submission:
            self.poisoned = True
        if self.poisoned:
            raise ValueError("proof_closed_or_poisoned")
        if submission and sum(map(len, (self.records, self.nodes, self.stream_generations, self.event_generations))) >= LIMIT:
            raise ValueError("proof_state_limit")

    @staticmethod
    def _new(key, generations, live):
        key = identity(key)
        if key[1] != generations.get(key[0], 0) + 1 or any(k[0] == key[0] for k in live):
            raise ValueError("proof_lifetime_generation")
        generations[key[0]] = key[1]
        return key

    def create_stream(self, key):
        self._admit()
        if len(self.stream_generations) >= 64:
            raise ValueError("proof_stream_limit")
        self.streams[self._new(key, self.stream_generations, self.streams)] = {}

    def create_event(self, key):
        self._admit()
        self.events[self._new(key, self.event_generations, self.events)] = 0

    def submit(self, stream):
        self._admit()
        stream = identity(stream)
        if stream not in self.streams:
            raise ValueError("proof_stale_stream")
        clock = self.streams[stream]
        cost = len(clock) + int(stream not in clock)
        if self.clock_entries + cost > 1_000_000:
            raise ValueError("proof_clock_limit")
        clock[stream] = clock.get(stream, 0) + 1
        node = len(self.nodes) + 1
        self.nodes[node] = dict(clock)
        self.clock_entries += cost
        return node

    def record(self, event, stream):
        self._admit()
        event = identity(event)
        if event not in self.events:
            raise ValueError("proof_stale_event")
        node = self.submit(stream)
        generation = self.events[event] + 1
        self.events[event] = generation
        token = (*event, generation)
        self.records[token] = self.nodes[node]  # immutable snapshot, not a second copy
        return token

    def _record(self, token, *, current=False):
        if type(token) is not tuple or len(token) != 3:
            raise ValueError("proof_event_record")
        for value in token:
            integer(value, positive=True)
        if token not in self.records or (current and self.events.get(token[:2]) != token[2]):
            raise ValueError("proof_stale_event_record")
        return self.records[token]

    def wait(self, stream, token):
        self._admit()
        stream = identity(stream)
        snapshot = self._record(token, current=True)
        if stream not in self.streams:
            raise ValueError("proof_stale_stream")
        cost = len(set(self.streams[stream]) | set(snapshot) | {stream})
        if self.clock_entries + cost > 1_000_000:
            raise ValueError("proof_clock_limit")
        for key, value in snapshot.items():
            self.streams[stream][key] = max(self.streams[stream].get(key, 0), value)
        return self.submit(stream)

    def query(self, token, result):
        self._admit(submission=False)
        snapshot = self._record(token, current=True)
        if result not in ("success", "not_ready"):
            self.poisoned = True
            raise ValueError("proof_event_query_failure")
        if result == "success":
            for key, value in snapshot.items():
                self.completed[key] = max(self.completed.get(key, 0), value)

    def _node(self, node):
        integer(node, positive=True)
        if node not in self.nodes:
            raise ValueError("proof_unknown_node")
        return self.nodes[node]

    def precedes(self, before, after):
        a, b = self._node(before), self._node(after)
        return all(value <= b.get(key, 0) for key, value in a.items())

    def retired(self, node):
        clock = self._node(node)
        return not self.poisoned and all(value <= self.completed.get(key, 0) for key, value in clock.items())

    def require_retired(self, users):
        if type(users) not in (list, tuple) or not users or len(users) > LIMIT:
            raise ValueError("proof_users_missing")
        if not all(self.retired(node) for node in users):
            raise ValueError("proof_early_reuse")

    def destroy_event(self, key):
        self._admit(submission=False)
        key = identity(key)
        if key not in self.events:
            raise ValueError("proof_stale_event")
        del self.events[key]  # submitted waits already own immutable snapshots

    def destroy_stream(self, key):
        self._admit(submission=False)
        key = identity(key)
        if key not in self.streams:
            raise ValueError("proof_stale_stream")
        if any(v > self.completed.get(k, 0) for k, v in self.streams[key].items()):
            raise ValueError("proof_stream_still_in_use")
        del self.streams[key]

    def seal_submissions(self):
        self._admit()
        self.sealed = True

    def terminal(self, *, producer_barrier, api_pairs_closed, flush_completed,
                 buffers_returned, records_consumed, dropped, errors):
        """Conditional model closure; caller booleans are NOT captured proof.

        A real collector needs a producer barrier BEFORE its final flush, outside
        callbacks/loader lock. Process exit or an empty buffer alone is insufficient.
        """
        facts = (producer_barrier, api_pairs_closed, flush_completed, buffers_returned, records_consumed)
        if any(type(v) is not bool for v in facts):
            raise ValueError("proof_terminal_boolean")
        integer(dropped); integer(errors)
        return (self.sealed and not self.poisoned and bool(self.nodes) and all(facts)
                and not dropped and not errors and all(self.retired(node) for node in self.nodes))
