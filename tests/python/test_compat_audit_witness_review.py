"""Independent counterexamples for the bounded native witness analyzers.

These intentionally require rejection of observations that cannot establish the
advertised single-producer ordering. No GPU, imported fixture, or OS API needed.
"""
import json
from pathlib import Path
import tempfile
import unittest

from xvram.compat_audit_order import analyze, analyze_rows
from xvram.compat_launch_census import CUPTI_SHA256


def _trace():
    rows = [dict(kind='session')]

    def pair(api, symbol, **fields):
        common = dict(api_id=api, thread_id=1, context_id=2, stream_id=0,
                      event_id=0, flags=0, stream_kind='none', symbol=symbol,
                      terminal_tool=False, result=0)
        common.update(fields)
        rows.extend([dict(common, kind='api_enter'),
                     dict(common, kind='api_exit')])

    pair(1, 'cuDevicePrimaryCtxRetain')
    pair(2, 'cuLaunchKernel', stream_id=3, stream_kind='per_thread')
    pair(3, 'cuStreamSynchronize', stream_id=3, stream_kind='per_thread')
    rows.append(dict(kind='seal', open_calls=0))
    pair(4, 'cuCtxSynchronize', terminal_tool=True)
    rows.append(dict(kind='summary', gpu_drained=True, census_drained=True,
                     sealed=True, errors=0, open_calls=0))
    return _number(rows)


def _number(rows):
    for sequence, row in enumerate(rows, 1):
        row.update(schema_version=1, record_type='xvram.cuda_order_witness',
                   sequence=sequence)
    return rows


def _census(rows, version=2):
    result = [dict(kind='session', terminal_complete=False, cupti_sha256=CUPTI_SHA256)]
    for row in rows:
        if row['kind'] not in ('api_enter', 'api_exit'):
            continue
        item = dict(kind=row['kind'], domain='driver', api_id=row['api_id'],
                    parent_api_id=0, correlation_id=row['api_id']*10,
                    probe_call_id=int(row['symbol'] == 'cuLaunchKernel'), symbol=row['symbol'])
        if row['kind'] == 'api_exit':
            item['result'] = row['result']
        result.append(item)
    result.extend([dict(kind='kernel', correlation_id=20, name='kernel', start_ns=1, end_ns=2),
                   dict(kind='summary', errors=0, dropped=0, buffers_outstanding=0, terminal_complete=False)])
    for sequence, row in enumerate(result, 1):
        row.update(schema_version=version, record_type='xvram.cuda_launch_census', sequence=sequence)
    return result


def _files_analyze(order, census):
    with tempfile.TemporaryDirectory() as directory:
        order_path, census_path = (Path(directory)/name for name in ('order.jsonl', 'census.jsonl'))
        order_path.write_text(''.join(json.dumps(row)+'\n' for row in order), encoding='utf-8')
        census_path.write_text(''.join(json.dumps(row)+'\n' for row in census), encoding='utf-8')
        return analyze(order_path, census_path)


class WitnessReviewTests(unittest.TestCase):
    def test_control_trace_is_valid(self):
        counts, _ = analyze_rows(_trace())
        self.assertEqual(counts['kernels'], 1)

    def test_reject_nested_mutations_without_submission_witness(self):
        rows = _trace()
        # Outer launch ENTER, inner launch ENTER/EXIT, outer launch EXIT.
        # API return order alone cannot prove which launch submitted first.
        rows[4:4] = [dict(rows[3], api_id=99), dict(rows[4], api_id=99)]
        with self.assertRaises(ValueError):
            analyze_rows(_number(rows))

    def test_reject_crossed_callbacks_on_one_producer(self):
        rows = _trace()
        # A1 ENTER, A2 ENTER, A1 EXIT, A2 EXIT is not a same-thread call stack.
        # Use queries, for which proper nesting is allowed, to exercise LIFO
        # separately from the overlapping-mutating-interval restriction.
        query = dict(rows[1], symbol='cuCtxGetCurrent')
        rows[3:3] = [dict(query, api_id=98, kind='api_enter'),
                     dict(query, api_id=99, kind='api_enter'),
                     dict(query, api_id=98, kind='api_exit'),
                     dict(query, api_id=99, kind='api_exit')]
        with self.assertRaises(ValueError):
            analyze_rows(_number(rows))

    def test_nested_query_does_not_create_a_submission_edge(self):
        rows = _trace()
        query = dict(rows[1], symbol='cuCtxGetCurrent', api_id=99)
        rows[4:4] = [dict(query, kind='api_enter'), dict(query, kind='api_exit')]
        counts, _ = analyze_rows(_number(rows))
        self.assertEqual(counts['kernels'], 1)

    def test_complete_census_versions(self):
        for version in (2, 3):
            with self.subTest(version=version):
                rows = _trace()
                result = _files_analyze(rows, _census(rows, version))
                self.assertEqual(result['counts']['kernels'], 1)
                self.assertFalse(result['memory_bounds'])
                self.assertFalse(result['cubin_binding'])

    def test_reject_unclean_census_counters(self):
        for field in ('errors', 'dropped', 'buffers_outstanding'):
            with self.subTest(field=field):
                rows = _trace()
                census = _census(rows)
                census[-1][field] = 1
                with self.assertRaises(ValueError):
                    _files_analyze(rows, census)

    def test_reject_open_census_api_outside_order_subset(self):
        rows = _trace()
        census = _census(rows)
        census.insert(-1, dict(schema_version=2, record_type='xvram.cuda_launch_census',
            kind='api_enter', api_id=5, parent_api_id=0, domain='driver',
            correlation_id=50, probe_call_id=0, symbol='cuMemAlloc_v2'))
        for sequence, row in enumerate(census, 1):
            row['sequence'] = sequence
        with self.assertRaises(ValueError):
            _files_analyze(rows, census)

    def test_reject_census_exits_without_enters_or_terminal(self):
        rows = _trace()
        # The public analyzer must not accept a file containing only exits as
        # independent census coverage, even when its surviving exit map matches.
        census = []
        for row in rows:
            if row['kind'] == 'api_exit':
                census.append(dict(schema_version=2,
                    record_type='xvram.cuda_launch_census', sequence=len(census)+1,
                    kind='api_exit', domain='driver', api_id=row['api_id'],
                    symbol=row['symbol'], result=row['result']))
        with self.assertRaises(ValueError):
            _files_analyze(rows, census)


if __name__ == '__main__':
    unittest.main()
