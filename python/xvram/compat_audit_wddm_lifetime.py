"""Offline WDDM lifetime graph over *normalized* observations, not raw ETW.

This is deliberately not an ETL/SQLite reader. A separate private normalizer
must verify providers, payload versions, timestamps, process ownership, and
loss counters before replacing PID/native handles with bounded process-local
IDs. Raw addresses, image names, and native handles are not accepted here.

Identity namespaces matter: the inspected pilot's HwQueue Stop zeroes the
driver hHwQueue field but retains ParentDxgHwQueue. The latter is a distinct
observed OS queue identity, NOT a parent_queue edge. A normalizer must bind
queue_id to the reviewed OS identity namespace and correlate it with hContext.
Likewise ContextHandle is zero at Context Stop; hContext is the observed key.
Optional normalized parent links below require their own evidence and must
never be manufactured by copying similarly named fields across namespaces.

Start/Stop pairing establishes an observed resource graph only. In particular,
the DxgKrnl Stop payload has no successful-DDI-return status. Neither balanced
resource lifetimes nor process reap prove callback-channel lifetime, actual
device retirement, or a one-to-one CUDA-to-WDDM context mapping.
"""
from .compat_audit_identity import _uint


MAX_EVENTS = 1_000_000
MAX_IDS = 100_000
COMMON = frozenset(('kind', 'sequence', 'timestamp_ticks'))
CAPTURE_FIELDS = frozenset((
    'start_ticks', 'end_ticks', 'reap_ticks', 'clock_frequency', 'events_lost',
    'log_buffers_lost', 'realtime_buffers_lost', 'decoder_errors', 'etl_closed',
    'trace_truncated', 'controller_reaped', 'process_tree_drained', 'timed_out',
    'worker_exit_code',
))
PROCESS_FIELDS = frozenset(('process_id', 'process_generation'))
DEVICE_FIELDS = frozenset(('device_id', 'device_generation'))
CONTEXT_FIELDS = frozenset(('context_id', 'context_generation'))
QUEUE_FIELDS = frozenset(('queue_id', 'queue_generation'))
PARENT_PROCESS = frozenset(('parent_process_id', 'parent_process_generation'))
PARENT_CONTEXT = frozenset(('parent_context_id', 'parent_context_generation'))
PARENT_QUEUE = frozenset(('parent_queue_id', 'parent_queue_generation'))
EVENT_FIELDS = {
    'process_start': PROCESS_FIELDS | PARENT_PROCESS,
    'process_stop': PROCESS_FIELDS | {'exit_code'},
    'device_start': PROCESS_FIELDS | DEVICE_FIELDS,
    'device_stop': PROCESS_FIELDS | DEVICE_FIELDS,
    'context_start': DEVICE_FIELDS | CONTEXT_FIELDS | PARENT_CONTEXT,
    'context_stop': DEVICE_FIELDS | CONTEXT_FIELDS | PARENT_CONTEXT,
    'queue_start': CONTEXT_FIELDS | QUEUE_FIELDS | PARENT_QUEUE,
    'queue_stop': CONTEXT_FIELDS | QUEUE_FIELDS | PARENT_QUEUE,
}


def _key(row, prefix, *, optional=False):
    value = (_uint(row[prefix + '_id'], 0 if optional else 1, MAX_IDS),
             _uint(row[prefix + '_generation'], 0 if optional else 1, MAX_IDS))
    if optional and (value[0] == 0) != (value[1] == 0):
        raise ValueError('wddm_partial_null_key')
    return value


def _fields(row, fields):
    if type(row) is not dict or set(row) != fields:
        raise ValueError('wddm_fields')


class _Registry:
    def __init__(self):
        self.objects = {}
        self.last = {}

    def start(self, key, timestamp, **data):
        previous = self.last.get(key[0])
        expected = 1 if previous is None else previous[1] + 1
        if key[1] != expected:
            raise ValueError('wddm_generation_gap_or_aba')
        if previous is not None:
            old = self.objects[previous]
            if old['stop'] is None:
                raise ValueError('wddm_reuse_live_resource')
            # Identical clock ticks do not establish a strict ABA boundary.
            if timestamp <= old['stop']:
                raise ValueError('wddm_ambiguous_reuse_boundary')
        self.objects[key] = dict(data, start=timestamp, stop=None)
        self.last[key[0]] = key
        return self.objects[key]

    def live(self, key):
        item = self.objects.get(key)
        if item is None or item['stop'] is not None:
            raise ValueError('wddm_unknown_or_stale_resource')
        return item

    def stop(self, key, timestamp):
        item = self.live(key)
        if timestamp <= item['start']:
            raise ValueError('wddm_nonpositive_lifetime')
        item['stop'] = timestamp

    def any_live(self, field=None, value=None):
        return any(item['stop'] is None and (field is None or item[field] == value)
                   for item in self.objects.values())


def analyze_lifetimes(events, capture, *, root_process):
    """Validate selected root/descendant resource observations, with false GO.

    The selected root's external parent is normalized to (0, 0); descendants
    must point to an observed live process generation. Other system processes
    must not be imported into this selected-process graph. A private normalizer
    must not drop an unresolved resource in order to manufacture that selection.

    Capture uses one verified monotonic clock. Collection must start before
    root birth and end after confirmed process-tree reap. Resource Stop records
    may arrive with timestamps after process exit/reap: they are retained and
    reported, not silently assumed to have occurred at process exit.
    """
    _fields(capture, CAPTURE_FIELDS)
    if type(root_process) not in (tuple, list) or len(root_process) != 2:
        raise ValueError('wddm_root_key')
    root = tuple(_uint(value, 1, MAX_IDS) for value in root_process)
    start = _uint(capture['start_ticks'])
    end = _uint(capture['end_ticks'], start + 1)
    reap = _uint(capture['reap_ticks'], start + 1, end - 1)
    _uint(capture['clock_frequency'], 1, 10**12)
    for field in ('events_lost', 'log_buffers_lost', 'realtime_buffers_lost', 'decoder_errors'):
        if _uint(capture[field]) != 0:
            raise ValueError('wddm_loss_or_decode_error')
    for field in ('etl_closed', 'controller_reaped', 'process_tree_drained'):
        if capture[field] is not True:
            raise ValueError('wddm_capture_not_closed_or_reaped')
    for field in ('trace_truncated', 'timed_out'):
        if capture[field] is not False:
            raise ValueError('wddm_capture_incomplete')
    if _uint(capture['worker_exit_code'], 0, 2**32 - 1) != 0:
        raise ValueError('wddm_worker_failed')

    processes, devices, contexts, queues = (_Registry() for _ in range(4))
    last_time = start
    count = 0
    for count, row in enumerate(events, 1):
        if count > MAX_EVENTS or type(row) is not dict:
            raise ValueError('wddm_event_capacity_or_type')
        kind = row.get('kind')
        if type(kind) is not str or kind not in EVENT_FIELDS:
            # Rundown (DCStart) is not creation. An unknown lifecycle kind is
            # evidence of incomplete coverage, not an ignorable annotation.
            raise ValueError('wddm_unknown_event_or_rundown')
        _fields(row, COMMON | EVENT_FIELDS[kind])
        if _uint(row['sequence'], 1, MAX_EVENTS) != count:
            raise ValueError('wddm_event_sequence')
        timestamp = _uint(row['timestamp_ticks'], start + 1, end - 1)
        if timestamp < last_time:
            raise ValueError('wddm_timestamp_inversion')
        last_time = timestamp
        if count == 1 and (kind != 'process_start' or _key(row, 'process') != root):
            raise ValueError('wddm_root_birth_missing')

        if kind == 'process_start':
            key = _key(row, 'process')
            parent = _key(row, 'parent_process', optional=True)
            if key == root:
                if count != 1 or parent != (0, 0):
                    raise ValueError('wddm_root_birth_or_parent')
            else:
                if parent == (0, 0) or parent == key:
                    raise ValueError('wddm_foreign_or_cyclic_process')
                processes.live(parent)
            processes.start(key, timestamp, parent=parent)
        elif kind == 'process_stop':
            key = _key(row, 'process')
            if timestamp > reap or _uint(row['exit_code'], 0, 2**32 - 1) != 0:
                raise ValueError('wddm_process_failure_or_after_reap')
            processes.stop(key, timestamp)
        elif kind == 'device_start':
            owner = _key(row, 'process')
            processes.live(owner)
            devices.start(_key(row, 'device'), timestamp, owner=owner)
        elif kind == 'device_stop':
            key, owner = _key(row, 'device'), _key(row, 'process')
            item = devices.live(key)
            if item['owner'] != owner:
                raise ValueError('wddm_device_owner_mismatch')
            if contexts.any_live('device', key):
                raise ValueError('wddm_device_stop_with_live_contexts')
            devices.stop(key, timestamp)
        elif kind in ('context_start', 'context_stop'):
            key, device = _key(row, 'context'), _key(row, 'device')
            parent = _key(row, 'parent_context', optional=True)
            device_row = devices.live(device)
            if kind == 'context_start':
                processes.live(device_row['owner'])
                if parent != (0, 0):
                    parent_row = contexts.live(parent)
                    if parent_row['owner'] != device_row['owner'] or parent == key:
                        raise ValueError('wddm_context_parent_mismatch')
                contexts.start(key, timestamp, device=device, parent=parent,
                               owner=device_row['owner'])
            else:
                item = contexts.live(key)
                if item['device'] != device or item['parent'] != parent:
                    raise ValueError('wddm_context_stop_identity')
                if queues.any_live('context', key) or contexts.any_live('parent', key):
                    raise ValueError('wddm_context_stop_with_live_children')
                contexts.stop(key, timestamp)
        else:
            key, context = _key(row, 'queue'), _key(row, 'context')
            parent = _key(row, 'parent_queue', optional=True)
            context_row = contexts.live(context)
            if kind == 'queue_start':
                processes.live(context_row['owner'])
                if parent != (0, 0):
                    parent_row = queues.live(parent)
                    if parent_row['owner'] != context_row['owner'] or parent == key:
                        raise ValueError('wddm_queue_parent_mismatch')
                queues.start(key, timestamp, context=context, parent=parent,
                             owner=context_row['owner'])
            else:
                item = queues.live(key)
                if item['context'] != context or item['parent'] != parent:
                    raise ValueError('wddm_queue_stop_identity')
                if queues.any_live('parent', key):
                    raise ValueError('wddm_queue_stop_with_live_children')
                queues.stop(key, timestamp)
    if (not count or root not in processes.objects or not devices.objects
            or not contexts.objects or not queues.objects
            or any(registry.any_live() for registry in (processes, devices, contexts, queues))):
        raise ValueError('wddm_incomplete_lifetime_graph')
    resources = list(devices.objects.values()) + list(contexts.objects.values()) + list(queues.objects.values())
    late = sum(item['stop'] > processes.objects[item['owner']]['stop'] for item in resources)
    return {
        'observation': 'normalized_wddm_lifetimes_balanced',
        'root_process': {'id': root[0], 'generation': root[1]},
        'counts': {'events': count, 'process_generations': len(processes.objects),
                   'device_generations': len(devices.objects),
                   'context_generations': len(contexts.objects),
                   'queue_generations': len(queues.objects),
                   'resource_stops_after_process_exit': late,
                   'resource_stops_after_reap': sum(item['stop'] > reap for item in resources)},
        'capture_encloses_birth_and_reap': True,
        'observed_lifetimes_balanced': True,
        'successful_resource_destruction_proven': False,
        'observer_lifetime_proven': False,
        'proof': dict.fromkeys(('memory_bounds', 'cubin_binding', 'device_ordering',
                               'trace_completeness'), False),
        'unresolved': ['private_raw_to_normalized_provenance',
                       'resource_stop_success_semantics',
                       'cupti_to_wddm_context_relation',
                       'observer_coverage_until_actual_resource_closure',
                       'native_gpu_retirement_and_cross_channel_correlation'],
    }
