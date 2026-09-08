"""Offline, typed native-teardown model; never a terminal-completeness proof.

Inputs are normalized observations, not a new accepted native trace format.
The census pairs must cover the whole capture, and typed lifecycle rows must
join their actual API instance IDs. A checkpoint is an execution-drain cutoff,
NOT a claim that callbacks, CUDA resources, or the process have ceased to exist.

Current historical post-footer census rows lack typed library generations and
producer tokens. They cannot satisfy this model by matching API names alone.
Even successful validation leaves observer lifetime, Activity delivery, actual
context destruction, and the four profile-wide proof gates unproved.
"""
from .compat_audit_identity import _uint
from .compat_launch_census import _name


LOADS = frozenset(('cuLibraryLoadData', 'cuLibraryLoadFromFile'))
RETAIN = 'cuDevicePrimaryCtxRetain'
RELEASES = frozenset(('cuDevicePrimaryCtxRelease', 'cuDevicePrimaryCtxRelease_v2'))
LIFECYCLE = LOADS | RELEASES | {RETAIN, 'cuLibraryUnload'}
PAIR_FIELDS = frozenset('api_id parent_api_id domain symbol result enter_sequence exit_sequence'.split())
TYPED_FIELDS = frozenset('api_id producer_id library_id library_generation context_id context_generation device'.split())
MAX_APIS = 2_000_000
MAX_OBJECTS = 100_000


def _fields(row, fields):
    if type(row) is not dict or set(row) != fields:
        raise ValueError('teardown_fields')


def analyze_epoch(api_pairs, typed_lifecycles, *, checkpoint_sequence):
    """Validate only observed destructor order and application-owned references.

    ``api_pairs`` contains every census API, ordered by actual instance ID,
    including non-lifecycle execution APIs before the checkpoint. Sequence
    numbers are from that same census. ``typed_lifecycles`` contains exactly
    its lifecycle subset, with monotonic observation IDs, never raw handles.

    This function does not accept a caller's ``context_destroyed=True`` or
    ``observer_complete=True`` assertion. No such evidence is present here.
    Callers must separately verify framing, provenance, reap, Activity loss,
    prefix device retirement and observer coverage through resource closure.
    """
    cutoff = _uint(checkpoint_sequence, 1, 4_000_000)
    lifecycle, tails, boundaries = {}, [], set()
    pair_count = 0
    previous_tail_end = cutoff
    previous_start = 0
    for pair_count, pair in enumerate(api_pairs, 1):
        _fields(pair, PAIR_FIELDS)
        api = _uint(pair['api_id'], 1, MAX_APIS)
        if api != pair_count:
            raise ValueError('teardown_census_api_sequence')
        parent = _uint(pair['parent_api_id'], 0, api - 1)
        start = _uint(pair['enter_sequence'], 1, 4_000_000)
        end = _uint(pair['exit_sequence'], start + 1, 4_000_000)
        if start <= previous_start:
            raise ValueError('teardown_census_enter_order')
        previous_start = start
        result = _uint(pair['result'], 0, 2**31 - 1)
        symbol = _name(pair['symbol'])
        if pair['domain'] not in ('driver', 'runtime'):
            raise ValueError('teardown_census_domain')
        if start in boundaries or end in boundaries or start == cutoff or end == cutoff:
            raise ValueError('teardown_census_boundary_alias')
        boundaries.update((start, end))
        if start < cutoff < end:
            raise ValueError('teardown_checkpoint_open_api')
        if start > cutoff:
            if (pair['domain'] != 'driver' or symbol not in RELEASES | {'cuLibraryUnload'}
                    or parent or result):
                raise ValueError('teardown_unknown_or_failed_tail_api')
            if start <= previous_tail_end:
                raise ValueError('teardown_overlapping_tail_producers')
            previous_tail_end = end
            tails.append(api)
        if symbol in LIFECYCLE:
            if pair['domain'] != 'driver' or result:
                raise ValueError('teardown_lifecycle_failure_or_domain')
            lifecycle[api] = pair
    if not tails or not lifecycle:
        raise ValueError('teardown_epoch_missing')

    libraries, contexts, device_contexts = {}, {}, {}
    producer = None
    seen = set()
    previous_api = 0
    tail_unloads = tail_releases = 0
    for row in typed_lifecycles:
        _fields(row, TYPED_FIELDS)
        api = _uint(row['api_id'], 1, MAX_APIS)
        if api <= previous_api or api not in lifecycle:
            raise ValueError('teardown_typed_api_join')
        previous_api = api
        token = _uint(row['producer_id'], 1, MAX_OBJECTS)
        if producer is not None and token != producer:
            raise ValueError('teardown_unknown_producer')
        producer = token
        lib = _uint(row['library_id'], 0, MAX_OBJECTS)
        libgen = _uint(row['library_generation'], 0, MAX_OBJECTS)
        context = _uint(row['context_id'], 0, MAX_OBJECTS)
        ctxgen = _uint(row['context_generation'], 0, MAX_OBJECTS)
        device = _uint(row['device'], 0, 127)
        pair = lifecycle[api]
        symbol = pair['symbol']
        tail = pair['enter_sequence'] > cutoff
        if symbol in LOADS | {'cuLibraryUnload'}:
            if not lib or not libgen or context or ctxgen or device:
                raise ValueError('teardown_library_fields')
            generation, live = libraries.get(lib, (0, False))
            if symbol in LOADS:
                if live or libgen != generation + 1:
                    raise ValueError('teardown_library_generation')
                libraries[lib] = (libgen, True)
            else:
                if not live or libgen != generation:
                    raise ValueError('teardown_stale_library_unload')
                libraries[lib] = (generation, False)
                tail_unloads += int(tail)
        else:
            if lib or libgen or not context or not ctxgen:
                raise ValueError('teardown_primary_fields')
            key = (context, ctxgen)
            # A zero application retain balance is NOT destruction. Reacquire
            # or generation replacement needs independent resource evidence.
            if device in device_contexts and device_contexts[device] != key:
                raise ValueError('teardown_primary_identity_changed')
            if contexts and key not in contexts:
                raise ValueError('teardown_multiple_or_reused_contexts')
            if key in contexts and contexts[key][0] != device:
                raise ValueError('teardown_context_device_changed')
            prior = contexts.get(key)
            if symbol == RETAIN:
                if prior is None:
                    if ctxgen != 1:
                        raise ValueError('teardown_unobserved_context_generation')
                    references = 0
                else:
                    references = prior[1]
                    if references == 0:
                        raise ValueError('teardown_retain_after_reference_cutoff')
                contexts[key] = (device, references + 1)
                device_contexts[device] = key
            else:
                if prior is None or prior[1] == 0:
                    raise ValueError('teardown_unmatched_primary_release')
                if tail and any(live for _, live in libraries.values()):
                    raise ValueError('teardown_primary_release_before_library_unloads')
                contexts[key] = (device, prior[1] - 1)
                tail_releases += int(tail)
        seen.add(api)
    if seen != set(lifecycle):
        raise ValueError('teardown_missing_typed_lifecycle')
    if (not tail_unloads or not tail_releases or any(live for _, live in libraries.values())
            or not contexts or any(refs for _, refs in contexts.values())):
        raise ValueError('teardown_owned_resources_not_balanced')
    return {
        'observation': 'typed_native_teardown_only',
        'counts': {'census_api_pairs': pair_count, 'typed_lifecycle_pairs': len(seen),
                   'tail_api_pairs': len(tails), 'tail_library_unloads': tail_unloads,
                   'tail_primary_releases': tail_releases},
        'observed_library_generations_balanced': True,
        'observed_application_primary_references_balanced': True,
        'context_destruction_proven': False,
        'observer_lifetime_proven': False,
        'proof': dict.fromkeys(('memory_bounds', 'cubin_binding', 'device_ordering',
                               'trace_completeness'), False),
        'unresolved': ['prefix_gpu_retirement', 'activity_disable_delivery_and_loss',
                       'actual_context_and_queue_resource_closure',
                       'observer_coverage_through_resource_closure',
                       'post_reap_counter_and_census_reconciliation'],
    }
