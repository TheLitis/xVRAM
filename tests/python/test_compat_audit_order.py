import copy
import unittest

from xvram.compat_audit_order import analyze_rows


def trace():
    rows = [dict(kind='session')]
    def pair(symbol, **fields):
        api = (len(rows)+1)//2
        common = dict(api_id=api, thread_id=1, context_id=2, stream_id=0, event_id=0,
                      flags=0, stream_kind='none', symbol=symbol, terminal_tool=False, result=0)
        common.update(fields)
        begin = dict(common, kind='api_enter')
        if symbol == 'cuStreamCreate':
            begin.update(stream_id=0, stream_kind='none')
        if symbol == 'cuEventCreate':
            begin.update(event_id=0)
        rows.extend([begin, dict(common, kind='api_exit')])
    pair('cuDevicePrimaryCtxRetain')
    pair('cuStreamCreate', stream_id=3, stream_kind='explicit', flags=1)
    pair('cuEventCreate', event_id=5)
    pair('cuMemcpyHtoDAsync_v2', stream_id=4, stream_kind='per_thread')
    pair('cuStreamSynchronize', stream_id=4, stream_kind='per_thread')
    pair('cuLaunchKernel', stream_id=3, stream_kind='explicit')
    pair('cuEventRecord', stream_id=3, stream_kind='explicit', event_id=5)
    pair('cuStreamWaitEvent', stream_id=4, stream_kind='per_thread', event_id=5)
    pair('cuEventQuery', event_id=5)
    pair('cuStreamSynchronize', stream_id=4, stream_kind='per_thread')
    pair('cuStreamDestroy_v2', stream_id=3, stream_kind='explicit')
    pair('cuEventDestroy_v2', event_id=5)
    rows.append(dict(kind='seal', open_calls=0))
    pair('cuCtxSynchronize', terminal_tool=True)
    rows.append(dict(kind='summary', gpu_drained=True, census_drained=True, sealed=True, errors=0, open_calls=0))
    for seq, row in enumerate(rows, 1):
        row.update(schema_version=1, record_type='xvram.cuda_order_witness', sequence=seq)
    return rows


class OrderTests(unittest.TestCase):
    def test_actual_order_and_retirement(self):
        counts, calls = analyze_rows(trace())
        self.assertEqual(counts['kernels'], 1)
        self.assertEqual(counts['event_records'], 1)
        self.assertEqual(counts['events_created'], counts['events_destroyed'])
        self.assertEqual(counts['api_pairs'], len(calls))

    def test_false_terminal_flags(self):
        for key, value in [('gpu_drained', False), ('census_drained', False), ('sealed', False),
                           ('errors', 1), ('open_calls', 1), ('errors', True)]:
            rows = trace()
            rows[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_rows(rows)

    def test_no_footer_or_late_producer(self):
        for rows in (trace()[:-1], trace()[1:], trace() + [trace()[-1]], trace() + [trace()[1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analyze_rows(rows)

    def test_rejects_second_thread_or_context(self):
        for field, value in [('thread_id', 8), ('context_id', 9), ('stream_id', 99),
                             ('stream_kind', 'legacy'), ('terminal_tool', True), ('result', 999)]:
            rows = trace()
            for row in rows:
                if row.get('symbol') == 'cuLaunchKernel':
                    row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_rows(rows)

    def test_no_unproved_api_or_raw_pointer(self):
        for field, value in [('symbol', 'cuGraphLaunch'), ('cuda_address', 1234), ('flags', -1), ('event_id', True)]:
            rows = trace()
            rows[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze_rows(rows)

    def test_generation_stale_event(self):
        rows = trace()
        for row in rows:
            if row.get('symbol') == 'cuStreamWaitEvent':
                row['event_id'] = 77
        with self.assertRaises(ValueError):
            analyze_rows(rows)

    def test_open_api_at_seal_and_duplicate(self):
        rows = trace()
        begin = copy.deepcopy(rows[11])
        for bad in (rows[:-4] + [begin] + rows[-4:], rows[:3] + rows[1:3] + rows[3:]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analyze_rows(bad)

    def test_wrong_wait_flags(self):
        rows = trace()
        for row in rows:
            if row.get('symbol') == 'cuStreamWaitEvent':
                row['flags'] = 1
        with self.assertRaises(ValueError):
            analyze_rows(rows)

    def test_early_destroy(self):
        rows = [r for r in trace() if r.get('symbol') not in ('cuEventQuery', 'cuStreamSynchronize')]
        with self.assertRaises(ValueError):
            analyze_rows(rows)


if __name__ == '__main__':
    unittest.main()
