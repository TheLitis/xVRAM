import copy
import json
from pathlib import Path
import tempfile
import unittest

from xvram.compat_audit_postmortem import FIELDS, FORMAT, MAGIC
from xvram.compat_audit_teardown_trace import analyze, load_sidecar
from xvram.compat_launch_census import CUPTI_SHA256


def fixture():
    census = [dict(kind='session', terminal_complete=False, cupti_sha256=CUPTI_SHA256)]
    def pair(api, symbol):
        common = dict(api_id=api, parent_api_id=0, domain='driver', correlation_id=100+api,
                      probe_call_id=1 if symbol == 'cuLaunchKernel' else 0, symbol=symbol)
        census.extend([dict(common, kind='api_enter'), dict(common, kind='api_exit', result=0)])
    pair(1, 'cuDevicePrimaryCtxRetain')
    pair(2, 'cuLibraryLoadData')
    pair(3, 'cuLaunchKernel')
    census.append(dict(kind='kernel', correlation_id=103, name='test_kernel', start_ns=10, end_ns=20))
    pair(4, 'cuCtxSynchronize')
    census.append(dict(kind='summary', errors=0, dropped=0, buffers_outstanding=0, terminal_complete=False))
    pair(5, 'cuLibraryUnload')
    pair(6, 'cuDevicePrimaryCtxRelease')
    typed = [dict(kind='session', qpc_ticks=100, qpc_frequency=10000000,
                  terminal_complete=False, context_destruction_proven=False)]
    def lifetime(api, symbol, *, library=False):
        typed.append(dict(kind='lifecycle', qpc_ticks=100+api*10, api_id=api, producer_id=1,
                          library_id=1 if library else 0, library_generation=1 if library else 0,
                          context_id=0 if library else 1, context_generation=0 if library else 1,
                          device=0, symbol=symbol, result=0, errors=0))
    lifetime(1, 'cuDevicePrimaryCtxRetain')
    lifetime(2, 'cuLibraryLoadData', library=True)
    typed.append(dict(kind='checkpoint', qpc_ticks=140, last_seen_driver_api_id=4,
                      lifecycle_pairs=2, pending_lifecycles=0, errors=0, terminal_complete=False))
    lifetime(5, 'cuLibraryUnload', library=True)
    lifetime(6, 'cuDevicePrimaryCtxRelease')
    for i, row in enumerate(census, 1):
        row.update(schema_version=3, record_type='xvram.cuda_launch_census', sequence=i)
    for i, row in enumerate(typed, 1):
        row.update(schema_version=1, record_type='xvram.cuda_teardown_witness', sequence=i)
    return census, typed


def ledger(**changes):
    row = dict.fromkeys(FIELDS, 0)
    row.update(magic=MAGIC, version=1, bytes=FORMAT.size, armed=1, sealed=1, finished=1,
               gpu_drained=1, activity_drained=1, census_closed=1, entered=6, exited=6, late_entries=2)
    row.update(changes)
    return FORMAT.pack(*(row[key] for key in FIELDS))


def capture(**changes):
    row = dict(controller_reaped=True, process_tree_drained=True, timed_out=False, exit_code=0, errors=[])
    row.update(changes)
    return row


class TeardownTraceTests(unittest.TestCase):
    def run_case(self, census=None, typed=None, *, metadata=None, postmortem=None):
        c, t = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, rows in [('census', c if census is None else census), ('typed', t if typed is None else typed)]:
                (root / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='ascii')
            return analyze(root / 'typed', root / 'census', 1,
                           capture() if metadata is None else metadata,
                           ledger() if postmortem is None else postmortem)

    def test_actual_checkpoint_and_typed_tail_reconcile_without_global_go(self):
        result = self.run_case()
        self.assertEqual(result['counts']['tail_api_pairs'], 2)
        self.assertTrue(result['observed_post_reap_counters_reconciled'])
        self.assertFalse(result['context_destruction_proven'])
        self.assertFalse(result['observer_lifetime_proven'])
        self.assertTrue(all(v is False for v in result['proof'].values()))

    def test_every_sticky_postmortem_channel_stays_fail_closed(self):
        for field in ('callbacks_open', 'apis_open', 'errors', 'activity_callbacks_open',
                      'activity_outstanding', 'late_activity'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_case(postmortem=ledger(**{field: 1}))

    def test_typed_tail_does_not_excuse_missing_or_extra_callbacks(self):
        for changes in ({'entered': 7}, {'exited': 5}, {'late_entries': 1}, {'late_entries': 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_case(postmortem=ledger(**changes))

    def test_reap_exit_and_timeout_required(self):
        for changes in ({'controller_reaped': False}, {'process_tree_drained': False},
                        {'timed_out': True}, {'exit_code': 27}, {'exit_code': True}, {'errors': ['failure']}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_case(metadata=capture(**changes))

    def test_all_native_error_and_closure_claims_rejected(self):
        for index, field, value in [(0, 'terminal_complete', True), (0, 'context_destruction_proven', True),
                                    (1, 'errors', 1), (1, 'result', 1), (3, 'errors', 1),
                                    (3, 'pending_lifecycles', 1), (3, 'terminal_complete', True)]:
            _, rows = fixture()
            rows[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_case(typed=rows)

    def test_checkpoint_cross_trace_api_and_pair_counts(self):
        for field, value in [('last_seen_driver_api_id', 3), ('lifecycle_pairs', 1)]:
            _, rows = fixture()
            rows[3][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_case(typed=rows)

    def test_missing_tail_row_not_inferred_from_census_name(self):
        _, rows = fixture()
        rows.pop(-2)
        for number, row in enumerate(rows, 1):
            row['sequence'] = number
        with self.assertRaises(ValueError):
            self.run_case(typed=rows)

    def test_raw_pointer_extra_fields_and_qpc_inversion(self):
        for field, value in [('address', 123), ('qpc_ticks', 99), ('producer_id', True)]:
            _, rows = fixture()
            rows[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_case(typed=rows)

    def test_unknown_tail_api_even_when_both_sources_claim_same_name(self):
        census, rows = fixture()
        for row in census:
            if row.get('api_id') == 6:
                row['symbol'] = 'cuLaunchKernel'
        rows[-1]['symbol'] = 'cuLaunchKernel'
        with self.assertRaises(ValueError):
            self.run_case(census=census, typed=rows)

    def test_activity_after_checkpoint_is_not_hidden_by_typed_destructors(self):
        census, _ = fixture()
        extra = copy.deepcopy(next(row for row in census if row['kind'] == 'kernel'))
        extra['sequence'] = len(census) + 1
        census.append(extra)
        with self.assertRaises(ValueError):
            self.run_case(census=census)

    def test_prefix_remains_validated_by_frozen_census_analyzer(self):
        for field, value in [('dropped', 1), ('errors', 1), ('buffers_outstanding', 1)]:
            census, _ = fixture()
            next(row for row in census if row['kind'] == 'summary')[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_case(census=census)

    def test_rerecords_and_error_records_never_form_final_summary(self):
        _, rows = fixture()
        rows[-1]['kind'] = 'error'
        with self.assertRaises(ValueError):
            self.run_case(typed=rows)
        _, rows = fixture()
        rows.append(dict(rows[3], sequence=len(rows)+1, qpc_ticks=180))
        with self.assertRaises(ValueError):
            self.run_case(typed=rows)

    def test_framing_duplicate_keys_and_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad'
            for value in ('{}', '{"sequence":1,"sequence":1}\n', 'x'*65537+'\n'):
                path.write_text(value, encoding='ascii')
                with self.subTest(value=value[:30]), self.assertRaises(ValueError):
                    load_sidecar(path)
