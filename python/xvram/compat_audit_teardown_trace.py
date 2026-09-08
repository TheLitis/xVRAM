"""Strict bridge for the non-final typed teardown sidecar and full census.

This observes late native destructors, not global terminal completeness. The
old census/order/identity contracts remain unchanged and still reject tails.
Here the single census summary is explicitly an execution checkpoint; only
typed, serial, generation-valid unload/release calls may follow it.
"""
import json
from pathlib import Path
import tempfile

from .compat_audit_identity import COMMON, _fields, _uint, digest, records
from .compat_audit_postmortem import FIELDS, FORMAT, MAGIC
from .compat_audit_teardown import LIFECYCLE, RELEASES, TYPED_FIELDS, analyze_epoch
from .compat_launch_census import analyze as analyze_census

SIDECAR_CAP = 64 * 1024 * 1024
CENSUS_CAP = 1024 * 1024 * 1024


def load_sidecar(path):
    checkpoint = None
    frequency = None
    typed = []
    calls = {}
    previous_ticks = 0
    previous_api = 0
    for row in records(path, 'xvram.cuda_teardown_witness', cap=SIDECAR_CAP, max_records=200000):
        _uint(row.get('qpc_ticks'))
        if row['qpc_ticks'] < previous_ticks:
            raise ValueError('teardown_qpc_inversion')
        previous_ticks = row['qpc_ticks']
        kind = row['kind']
        if row['sequence'] == 1 and kind != 'session':
            raise ValueError('teardown_session_missing')
        if kind == 'session':
            _fields(row, 'qpc_ticks qpc_frequency terminal_complete context_destruction_proven')
            if row['sequence'] != 1 or row['terminal_complete'] is not False or row['context_destruction_proven'] is not False:
                raise ValueError('teardown_invalid_session_claim')
            frequency = _uint(row['qpc_frequency'], 1, 10**12)
        elif kind == 'checkpoint':
            _fields(row, 'qpc_ticks last_seen_driver_api_id lifecycle_pairs pending_lifecycles errors terminal_complete')
            if (checkpoint is not None or _uint(row['pending_lifecycles']) or _uint(row['errors'])
                    or row['terminal_complete'] is not False
                    or _uint(row['lifecycle_pairs']) != len(typed)
                    or _uint(row['last_seen_driver_api_id'], 1, 2000000) < previous_api):
                raise ValueError('teardown_checkpoint_invalid')
            checkpoint = row
        elif kind == 'lifecycle':
            if set(row) != COMMON | TYPED_FIELDS | {'qpc_ticks', 'symbol', 'result', 'errors'}:
                raise ValueError('teardown_lifecycle_fields')
            api = _uint(row['api_id'], 1, 2000000)
            if (api <= previous_api or row['symbol'] not in LIFECYCLE
                    or _uint(row['result'], 0, 2**31 - 1) or _uint(row['errors'])):
                raise ValueError('teardown_lifecycle_failure')
            if checkpoint is not None and (api <= checkpoint['last_seen_driver_api_id']
                                           or row['symbol'] not in RELEASES | {'cuLibraryUnload'}):
                raise ValueError('teardown_invalid_late_lifecycle')
            previous_api = api
            calls[api] = row['symbol'], row['result']
            for field in TYPED_FIELDS - {'api_id'}:
                _uint(row[field], 0, 127 if field == 'device' else 100000)
            typed.append({field: row[field] for field in TYPED_FIELDS})
        else:
            raise ValueError('teardown_error_or_unknown_record')
    if (frequency is None or checkpoint is None or not typed
            or previous_api <= checkpoint['last_seen_driver_api_id']):
        raise ValueError('teardown_sidecar_incomplete')
    return {'qpc_frequency': frequency, 'checkpoint': checkpoint, 'typed': typed, 'calls': calls}


def _partition_census(path, probe_calls):
    calls, pending = {}, {}
    checkpoint = None
    last_driver = 0
    prefix_last_driver = None
    with tempfile.TemporaryDirectory(prefix='xvram-teardown-prefix-') as temporary:
        prefix_path = Path(temporary) / 'census-prefix.jsonl'
        with prefix_path.open('w', encoding='ascii', newline='\n') as prefix:
            for row in records(path, 'xvram.cuda_launch_census', cap=CENSUS_CAP,
                               versions=(3,), max_records=4000000):
                before_checkpoint = checkpoint is None
                if before_checkpoint:
                    prefix.write(json.dumps(row, ensure_ascii=True, separators=(',', ':')) + '\n')
                elif row['kind'] not in ('api_enter', 'api_exit'):
                    raise ValueError('teardown_census_unknown_tail_record')
                if row['kind'] == 'summary':
                    checkpoint = row
                    prefix_last_driver = last_driver
                    if pending:
                        raise ValueError('teardown_census_open_checkpoint')
                    continue
                if row['kind'] not in ('api_enter', 'api_exit'):
                    continue
                fields = 'api_id parent_api_id domain correlation_id probe_call_id symbol'
                _fields(row, fields + (' result' if row['kind'] == 'api_exit' else ''))
                api = _uint(row['api_id'], 1, 2000000)
                if row['domain'] not in ('driver', 'runtime'):
                    raise ValueError('teardown_census_domain')
                if row['domain'] == 'driver':
                    last_driver = max(last_driver, api)
                if not before_checkpoint and (row['domain'] != 'driver' or row['parent_api_id'] != 0
                        or row['probe_call_id'] != 0 or row['symbol'] not in RELEASES | {'cuLibraryUnload'}):
                    raise ValueError('teardown_census_unexpected_tail_api')
                if row['kind'] == 'api_enter':
                    if api != len(calls) + len(pending) + 1 or api in calls or api in pending:
                        raise ValueError('teardown_census_api_sequence')
                    pending[api] = row
                else:
                    entered = pending.pop(api, None)
                    if entered is None or any(entered[key] != row[key] for key in fields.split()):
                        raise ValueError('teardown_census_exit_mismatch')
                    result = _uint(row['result'], 0, 2**31 - 1)
                    if not before_checkpoint and result:
                        raise ValueError('teardown_census_failed_tail')
                    calls[api] = dict(api_id=api, parent_api_id=row['parent_api_id'], domain=row['domain'],
                                      symbol=row['symbol'], result=result,
                                      enter_sequence=entered['sequence'], exit_sequence=row['sequence'])
        if checkpoint is None or pending:
            raise ValueError('teardown_census_incomplete')
        prefix_result = analyze_census(prefix_path, probe_calls)
        counts = prefix_result['counts']
        if counts['open_api_calls'] or counts['buffers_outstanding']:
            raise ValueError('teardown_prefix_not_drained')
    return calls, checkpoint, prefix_last_driver, counts


def _postmortem(data, capture, total, tail, sampled):
    if type(sampled) is not bool or type(data) is not bytes or len(data) != FORMAT.size:
        raise ValueError('teardown_postmortem_input')
    if (capture.get('controller_reaped') is not True or capture.get('process_tree_drained') is not True
            or capture.get('timed_out') is not False or type(capture.get('exit_code')) is not int
            or capture['exit_code'] != 0 or capture.get('errors') != []):
        raise ValueError('teardown_worker_not_reaped')
    row = dict(zip(FIELDS, FORMAT.unpack(data)))
    if (row['magic'], row['version'], row['bytes']) != (MAGIC, 1, FORMAT.size):
        raise ValueError('teardown_postmortem_header')
    for field in ('armed', 'sealed', 'finished', 'gpu_drained', 'activity_drained', 'census_closed'):
        if row[field] != 1:
            raise ValueError('teardown_postmortem_unfinished')
    for field in ('callbacks_open', 'apis_open', 'errors', 'activity_callbacks_open',
                  'activity_outstanding', 'late_activity'):
        if row[field] != 0:
            raise ValueError('teardown_postmortem_error_or_open')
    if not 0 < row['entered'] == row['exited'] == total <= 2000000 or row['late_entries'] != tail:
        raise ValueError('teardown_postmortem_census_mismatch')
    if (not 0 <= row['retained_names'] <= 65536 or row['sampler_disabled'] != int(sampled)
            or (not sampled and row['retained_names'])):
        raise ValueError('teardown_postmortem_sampler')


def analyze(path, census_path, probe_calls, capture, postmortem_data, *, sampled=False):
    """Require matching immutable files and a post-reap controller ledger.

    Unlike the old strict postmortem decoder, this permits exactly the census-
    enumerated typed destructor tail. It does NOT suppress late Activity,
    missing rows, unknown producers, or errors, and cannot return global GO.
    """
    before = digest(path, SIDECAR_CAP), digest(census_path, CENSUS_CAP)
    sidecar = load_sidecar(path)
    calls, cutoff, last_driver, prefix_counts = _partition_census(census_path, probe_calls)
    if sidecar['checkpoint']['last_seen_driver_api_id'] != last_driver:
        raise ValueError('teardown_cross_trace_checkpoint_mismatch')
    expected = {api: (row['symbol'], row['result']) for api, row in calls.items()
                if row['domain'] == 'driver' and row['symbol'] in LIFECYCLE}
    if sidecar['calls'] != expected:
        raise ValueError('teardown_cross_trace_lifecycle_mismatch')
    result = analyze_epoch((calls[api] for api in sorted(calls)), sidecar['typed'],
                           checkpoint_sequence=cutoff['sequence'])
    tail = result['counts']['tail_api_pairs']
    if result['counts']['census_api_pairs'] != prefix_counts['api_pairs'] + tail:
        raise ValueError('teardown_prefix_tail_reconciliation')
    _postmortem(postmortem_data, capture, len(calls), tail, sampled)
    if before != (digest(path, SIDECAR_CAP), digest(census_path, CENSUS_CAP)):
        raise ValueError('teardown_trace_changed')
    result['sha256'], result['census_sha256'] = before
    result['qpc_frequency'] = sidecar['qpc_frequency']
    result['observed_post_reap_counters_reconciled'] = True
    result['unresolved'].remove('post_reap_counter_and_census_reconciliation')
    return result
