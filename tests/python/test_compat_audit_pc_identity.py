import copy
import unittest

from xvram.compat_audit_pc_identity import analyze_rows, correlate


def trace(sampled=True):
    rows = [dict(kind='session', serialized_diagnostic=True, terminal_complete=False,
                 sampling_period=11, hardware_buffer_bytes=512*1024*1024,
                 configured_buffer_pcs=4096, result_buffer_pcs=4096),
            dict(kind='preflight', context_present=True, errors=0, copies_submitted=0),
            dict(kind='enabled', stall_reasons=38),
            dict(kind='cubin', cubin_cookie=12, bytes=100, sha256='a'*64)]
    if sampled:
        rows.append(dict(kind='sample_identity', observed_after_call=1, correlation_id=42,
                         cubin_cookie=12, function_index=3, name='kernel'))
    rows.extend([dict(kind='collection', observed_after_call=1, pc_records=int(sampled), dropped_samples=0),
                 dict(kind='summary', errors=0, copies_submitted=1, copies_retired=1,
                      terminal_complete=False, sampler_cleanup_proven=False)])
    for sequence, row in enumerate(rows, 1):
        row.update(schema_version=1, record_type='xvram.cuda_pc_witness', sequence=sequence)
    return rows


def apis():
    common = dict(domain='driver', symbol='cuLaunchKernel', probe_call_id=1, api_id=7, correlation_id=42)
    return [dict(common, kind='api_enter'), dict(common, kind='api_exit', result=0)]


class PcIdentityTests(unittest.TestCase):
    def test_sampled_binding_uses_actual_instance(self):
        counts, samples = analyze_rows(trace())
        self.assertEqual(counts['sampled_calls'], 1)
        self.assertEqual(correlate(samples, apis(), {1: 'kernel'}),
                         [dict(call_id=1, cubin_sha256='a'*64, function_index=3, name='kernel')])

    def test_empty_is_explicitly_uncovered(self):
        counts, samples = analyze_rows(trace(False))
        self.assertEqual(counts['unsampled_calls'], 1)
        self.assertEqual(correlate(samples, apis(), {1: 'kernel'}), [])

    def test_no_name_only_join(self):
        _, samples = analyze_rows(trace())
        calls = apis()
        for row in calls:
            row['correlation_id'] = 43
        with self.assertRaises(ValueError):
            correlate(samples, calls, {1: 'kernel'})

    def test_wrong_name_and_missing_call(self):
        _, samples = analyze_rows(trace())
        for names in ({1: 'other'}, {}, {1: 'kernel', 2: 'extra'}):
            with self.subTest(names=names), self.assertRaises(ValueError):
                correlate(samples, apis(), names)

    def test_failed_or_missing_native_exit(self):
        _, samples = analyze_rows(trace())
        bad = apis()
        bad[-1]['result'] = 999
        for calls in (bad, apis()[:1], apis() + [apis()[-1]], apis() + apis()):
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                correlate(samples, calls, {1: 'kernel'})

    def test_rejects_corruption_and_false_claims(self):
        changes = [(0, 'serialized_diagnostic', False), (0, 'terminal_complete', True),
                   (1, 'context_present', False), (1, 'errors', 1), (2, 'stall_reasons', 129),
                   (3, 'bytes', 2**64), (3, 'sha256', 'z'*64), (3, 'cubin_cookie', 0),
                   (4, 'observed_after_call', 2), (4, 'correlation_id', True),
                   (4, 'cubin_cookie', 999), (4, 'name', 'x\n'), (4, 'name', 'a'*4097),
                   (5, 'pc_records', 0), (5, 'dropped_samples', 1),
                   (6, 'errors', 1), (6, 'copies_retired', 0),
                   (6, 'terminal_complete', True), (6, 'sampler_cleanup_proven', 1),
                   (4, 'cuda_address', 1234)]
        for index, key, value in changes:
            rows = trace()
            rows[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                analyze_rows(rows)

    def test_lifetime_and_partial_trace_rejections(self):
        rows = trace()
        for bad in (rows[:-1], rows[1:], rows + [rows[-1]], rows[:5] + rows[4:],
                    rows[:2] + rows[1:], rows[:3] + rows[2:]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analyze_rows(bad)

    def test_crc_collision_and_ambiguity(self):
        rows = trace()
        collision = copy.deepcopy(rows[3])
        collision['sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            analyze_rows(rows[:4] + [collision] + rows[4:])
        ambiguous = copy.deepcopy(rows[4])
        ambiguous['function_index'] += 1
        rows[5]['pc_records'] = 2
        with self.assertRaises(ValueError):
            analyze_rows(rows[:5] + [ambiguous] + rows[5:])

    def test_native_error_rejected(self):
        rows = trace()
        rows[-1]['kind'] = 'error'
        with self.assertRaises(ValueError):
            analyze_rows(rows)


if __name__ == '__main__':
    unittest.main()
