"""Reconstruct observed single-producer, single-context CUDA order.

No name-based cubin binding, tensor bounds or universal execution proof is
inferred. Event/stream generations are reconstructed from actual typed APIs.
"""
from collections import Counter
from pathlib import Path

from .compat_audit_identity import records, digest, _fields, _uint
from .compat_launch_census import analyze as analyze_census
from .compat_audit_proof import DeviceOrder

ORDER_CAP = 256 * 1024 * 1024
CENSUS_CAP = 1024 * 1024 * 1024
MAX_RECORDS = 4_000_000
MAX_API_ID = 2_000_000

SUBMISSIONS = {'cuLaunchKernel', 'cuMemcpyHtoDAsync_v2', 'cuMemcpyDtoHAsync_v2',
               'cuMemcpyDtoDAsync_v2', 'cuMemsetD8Async'}
QUERIES = {'cuCtxGetCurrent', 'cuCtxGetDevice_v2', 'cuCtxGetLimit',
           'cuStreamGetCtx', 'cuStreamGetGreenCtx'}
MUTATIONS = SUBMISSIONS | {'cuDevicePrimaryCtxRetain', 'cuCtxSetCurrent',
    'cuCtxSynchronize', 'cuCtxSynchronize_v2', 'cuStreamCreate', 'cuStreamDestroy_v2',
    'cuStreamSynchronize', 'cuEventCreate', 'cuEventDestroy_v2', 'cuEventRecord',
    'cuEventQuery', 'cuEventSynchronize', 'cuStreamWaitEvent'}
API_FIELDS = 'api_id thread_id context_id stream_id event_id flags stream_kind symbol terminal_tool result'


def analyze_rows(rows):
    order = DeviceOrder()
    streams, events, stream_generations, event_generations = {}, {}, Counter(), Counter()
    pending, calls, producer_threads, contexts = {}, {}, set(), set()
    callback_stack = []
    counts = Counter(api_pairs=0, submissions=0, kernels=0, copies=0, stream_syncs=0,
                     context_syncs=0, event_records=0, event_waits=0, event_queries=0,
                     streams_created=0, streams_destroyed=0, events_created=0, events_destroyed=0)
    sealed = False
    summary = None
    active_context = None

    def stream(row):
        key, kind = row['stream_id'], row['stream_kind']
        if not key or kind not in ('explicit', 'per_thread', 'legacy'):
            raise ValueError('order_stream_missing')
        if key not in streams:
            # This bounded profile needs no legacy-default implicit ordering.
            # A per-thread stream is legal only with one proved producer.
            if kind != 'per_thread' or key in stream_generations:
                raise ValueError('order_uncreated_or_unsupported_stream')
            stream_generations[key] += 1
            order.create_stream((key, stream_generations[key]))
            streams[key] = kind, stream_generations[key]
        if streams[key][0] != kind:
            raise ValueError('order_stream_kind_changed')
        return key, streams[key][1]

    def event(row):
        key = row['event_id']
        if key not in events:
            raise ValueError('order_uncreated_event')
        return key, events[key]

    for number, row in enumerate(rows, 1):
        kind = row['kind']
        if summary is not None:
            raise ValueError('order_after_terminal')
        if number == 1 and kind != 'session':
            raise ValueError('order_session_missing')
        if kind == 'session':
            _fields(row, '')
            if number != 1:
                raise ValueError('order_duplicate_session')
            continue
        if kind == 'seal':
            _fields(row, 'open_calls')
            if sealed or pending or _uint(row['open_calls']):
                raise ValueError('order_seal_with_producers')
            sealed = True
            order.seal_submissions()
            continue
        if kind == 'summary':
            _fields(row, 'gpu_drained census_drained sealed errors open_calls')
            if (not sealed or pending or events or _uint(row['errors']) or _uint(row['open_calls'])
                    or any(row[k] is not True for k in ('gpu_drained', 'census_drained', 'sealed'))
                    or any(k == 'explicit' for k, _ in streams.values())
                    or not counts['kernels'] or not all(order.retired(n) for n in order.nodes)):
                raise ValueError('order_terminal_unproven')
            summary = row
            continue
        if kind not in ('api_enter', 'api_exit'):
            raise ValueError('order_error_or_unknown_record')
        _fields(row, API_FIELDS)
        api = _uint(row['api_id'], 1, MAX_API_ID)
        for field in ('thread_id', 'context_id', 'stream_id', 'event_id'):
            _uint(row[field], 1 if field == 'thread_id' else 0, 100000)
        _uint(row['flags'], maximum=2**32-1)
        _uint(row['result'], maximum=2**31-1)
        if type(row['terminal_tool']) is not bool or row['stream_kind'] not in ('none', 'explicit', 'per_thread', 'legacy'):
            raise ValueError('order_api_types')
        symbol = row['symbol']
        if symbol not in QUERIES | MUTATIONS:
            raise ValueError('order_unsupported_native_api')
        if row['terminal_tool']:
            if not sealed or symbol != 'cuCtxSynchronize':
                raise ValueError('order_unexpected_terminal_tool_api')
        elif sealed:
            raise ValueError('order_late_producer')
        producer_threads.add(row['thread_id'])
        if len(producer_threads) != 1:
            raise ValueError('order_multiple_producers')
        if row['context_id']:
            contexts.add(row['context_id'])
            if len(contexts) != 1:
                raise ValueError('order_multiple_contexts')
        if kind == 'api_enter':
            if api in calls or api in pending or row['result']:
                raise ValueError('order_duplicate_or_invalid_enter')
            # Nested mutations have overlapping submission intervals. Their API
            # EXIT order alone cannot establish actual GPU submission order.
            if symbol in MUTATIONS and any(item['symbol'] in MUTATIONS for item in pending.values()):
                raise ValueError('order_overlapping_mutations')
            if len(callback_stack) >= 64:
                raise ValueError('order_callback_depth')
            pending[api] = row
            callback_stack.append(api)
            continue
        # This profile admits exactly one CPU producer, so observed callback
        # intervals must form one properly nested call stack, including queries.
        if not callback_stack or callback_stack[-1] != api:
            raise ValueError('order_callback_stack_mismatch')
        callback_stack.pop()
        begin = pending.pop(api, None)
        changing = {'cuStreamCreate': {'stream_id', 'stream_kind'}, 'cuEventCreate': {'event_id'},
                    'cuDevicePrimaryCtxRetain': {'context_id'}, 'cuStreamGetGreenCtx': {'flags'}}.get(symbol, set())
        stable = set(API_FIELDS.split()) - {'result'} - changing
        if begin is None or any(begin[k] != row[k] for k in stable):
            raise ValueError('order_api_pair_mismatch')
        calls[api] = symbol, row['result']
        counts['api_pairs'] += 1
        if symbol in QUERIES:
            if symbol == 'cuStreamGetGreenCtx' and row['flags'] != 0:
                raise ValueError('order_green_context_unsupported')
            if row['result'] and not (symbol == 'cuCtxGetDevice_v2' and row['context_id'] == 0
                                      and active_context is None and row['result'] == 201):
                raise ValueError('order_query_failure')
            continue
        if row['result'] and not (symbol == 'cuEventQuery' and row['result'] == 600):
            raise ValueError('order_native_failure')
        if symbol == 'cuDevicePrimaryCtxRetain':
            if active_context is not None or not row['context_id']:
                raise ValueError('order_context_admission')
            active_context = row['context_id']
            continue
        if symbol == 'cuCtxSetCurrent':
            if (active_context is not None and row['context_id'] != active_context) or counts['submissions']:
                raise ValueError('order_context_switch')
            continue
        if active_context is None or row['context_id'] != active_context:
            raise ValueError('order_wrong_context')
        if symbol == 'cuStreamCreate':
            key = row['stream_id']
            if not key or key in streams or row['stream_kind'] != 'explicit' or row['flags'] != 1:
                raise ValueError('order_stream_creation')
            stream_generations[key] += 1
            order.create_stream((key, stream_generations[key]))
            streams[key] = 'explicit', stream_generations[key]
            counts['streams_created'] += 1
        elif symbol == 'cuStreamDestroy_v2':
            order.destroy_stream(stream(row)); del streams[row['stream_id']]
            counts['streams_destroyed'] += 1
        elif symbol in SUBMISSIONS:
            target = stream(row)
            order.inherit_host_retirements(target)
            order.submit(target)
            counts['submissions'] += 1
            counts['kernels' if symbol == 'cuLaunchKernel' else 'copies'] += 1
        elif symbol == 'cuStreamSynchronize':
            order.synchronize_stream(stream(row), 'success'); counts['stream_syncs'] += 1
        elif symbol in ('cuCtxSynchronize', 'cuCtxSynchronize_v2'):
            order.synchronize_context('success'); counts['context_syncs'] += 1
        elif symbol == 'cuEventCreate':
            key = row['event_id']
            if not key or key in events:
                raise ValueError('order_event_creation')
            event_generations[key] += 1; events[key] = event_generations[key]
            order.create_event((key, events[key])); counts['events_created'] += 1
        elif symbol == 'cuEventDestroy_v2':
            order.destroy_event(event(row)); del events[row['event_id']]
            counts['events_destroyed'] += 1
        elif symbol == 'cuEventRecord':
            target = stream(row); order.inherit_host_retirements(target)
            order.record(event(row), target); counts['event_records'] += 1
        elif symbol in ('cuEventQuery', 'cuEventSynchronize'):
            target = event(row)
            token = (*target, order.events[target])
            order.query(token, 'success' if row['result'] == 0 else 'not_ready')
            counts['event_queries'] += 1
        elif symbol == 'cuStreamWaitEvent':
            if row['flags'] != 0:
                raise ValueError('order_wait_flags')
            target, ev = stream(row), event(row)
            order.inherit_host_retirements(target)
            order.wait(target, (*ev, order.events[ev])); counts['event_waits'] += 1
    if summary is None:
        raise ValueError('order_terminal_missing')
    return dict(counts), calls


def analyze(path, census_path):
    before = digest(path, ORDER_CAP), digest(census_path, CENSUS_CAP)
    counts, calls = analyze_rows(records(path, 'xvram.cuda_order_witness',
                                       cap=ORDER_CAP, max_records=MAX_RECORDS))
    # A surviving exit map is not independent coverage evidence: the complete
    # census must first validate its own framing, API lifetimes, activity joins,
    # and terminal delivery counters. Its frozen report keeps proof flags false.
    census = analyze_census(Path(census_path), counts['kernels'])
    observed_counts = census['counts']
    if (census['sha256'] != before[1]
            or any(observed_counts[name] for name in ('open_api_calls', 'buffers_outstanding',
                                                      'uncorrelated_kernels', 'unmarked_kernels'))
            or any(observed_counts[name] != counts['kernels'] for name in
                   ('gpu_kernels_observed', 'marked_kernels', 'probe_calls_with_gpu_activity'))):
        raise ValueError('order_census_incomplete')
    observed = {}
    for row in records(census_path, 'xvram.cuda_launch_census', cap=CENSUS_CAP,
                       versions=(2, 3), max_records=MAX_RECORDS):
        if row['kind'] != 'api_exit' or row['domain'] != 'driver':
            continue
        symbol = row['symbol']
        if (symbol.startswith(('cuLaunch', 'cuMemcpy', 'cuMemset', 'cuStream', 'cuEvent', 'cuCtx', 'cuDevicePrimaryCtx', 'cuGraph'))
                or symbol in ('cuMemAllocAsync', 'cuMemFreeAsync')):
            if row['api_id'] in observed:
                raise ValueError('order_census_duplicate')
            observed[row['api_id']] = symbol, row['result']
    if calls != observed:
        raise ValueError('order_census_coverage_mismatch')
    if before != (digest(path, ORDER_CAP), digest(census_path, CENSUS_CAP)):
        raise ValueError('order_inputs_changed')
    return dict(counts=counts, provenance=dict(order_trace=before[0], census_trace=before[1]),
                scope='observed_single_producer_single_context',
                memory_bounds=False, cubin_binding=False)
