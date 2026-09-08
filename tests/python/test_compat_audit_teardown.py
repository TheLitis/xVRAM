import copy
import unittest

from xvram.compat_audit_teardown import analyze_epoch


def fixture():
    pairs, typed = [], []

    def pair(symbol, start, **fields):
        api = len(pairs) + 1
        pairs.append(dict(api_id=api, parent_api_id=0, domain='driver', symbol=symbol,
                          result=0, enter_sequence=start, exit_sequence=start + 1))
        if fields:
            row = dict(api_id=api, producer_id=1, library_id=0, library_generation=0,
                       context_id=0, context_generation=0, device=0)
            row.update(fields)
            typed.append(row)

    pair('cuDevicePrimaryCtxRetain', 2, context_id=2, context_generation=1)
    pair('cuLibraryLoadData', 4, library_id=1, library_generation=1)
    pair('cuLaunchKernel', 6)
    pair('cuCtxSynchronize', 8)
    pair('cuLibraryUnload', 11, library_id=1, library_generation=1)
    pair('cuDevicePrimaryCtxRelease', 13, context_id=2, context_generation=1)
    return pairs, typed


class TeardownTests(unittest.TestCase):
    def check(self, pairs, typed):
        return analyze_epoch(pairs, typed, checkpoint_sequence=10)

    def test_observation_never_implies_context_or_global_closure(self):
        result = self.check(*fixture())
        self.assertEqual(result['counts']['tail_api_pairs'], 2)
        self.assertTrue(result['observed_application_primary_references_balanced'])
        self.assertFalse(result['context_destruction_proven'])
        self.assertFalse(result['observer_lifetime_proven'])
        self.assertTrue(all(value is False for value in result['proof'].values()))
        self.assertIn('actual_context_and_queue_resource_closure', result['unresolved'])

    def test_name_only_tail_is_not_typed_evidence(self):
        pairs, typed = fixture()
        with self.assertRaisesRegex(ValueError, 'missing_typed'):
            self.check(pairs, typed[:2])

    def test_unknown_tail_producers_or_failures(self):
        for field, value in [('symbol', 'cuLaunchKernel'), ('symbol', 'cuMemFree_v2'),
                             ('symbol', 'cuCtxSynchronize'), ('result', 201),
                             ('domain', 'runtime'), ('parent_api_id', 1)]:
            pairs, typed = fixture()
            pairs[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check(pairs, typed)

    def test_different_observed_producer(self):
        pairs, typed = fixture()
        typed[-1]['producer_id'] = 2
        with self.assertRaisesRegex(ValueError, 'unknown_producer'):
            self.check(pairs, typed)

    def test_stale_library_or_context_generation(self):
        for index, field in [(-2, 'library_generation'), (-1, 'context_generation'),
                             (-2, 'library_id'), (-1, 'context_id')]:
            pairs, typed = fixture()
            typed[index][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(pairs, typed)

    def test_unmatched_release_and_duplicate_unload(self):
        pairs, typed = fixture()
        pairs[0]['symbol'] = 'cuDevicePrimaryCtxRelease'
        with self.assertRaisesRegex(ValueError, 'unmatched_primary_release'):
            self.check(pairs, typed)
        pairs, typed = fixture()
        pairs[1]['symbol'] = 'cuLibraryUnload'
        with self.assertRaisesRegex(ValueError, 'stale_library_unload'):
            self.check(pairs, typed)

    def test_reference_balance_is_not_inferred_from_last_release(self):
        pairs, typed = fixture()
        pairs[2]['symbol'] = 'cuDevicePrimaryCtxRetain'
        another = dict(typed[0], api_id=3)
        typed.insert(2, another)
        with self.assertRaisesRegex(ValueError, 'not_balanced'):
            self.check(pairs, typed)

    def test_checkpoint_must_have_no_open_api(self):
        pairs, typed = fixture()
        pairs[3]['exit_sequence'] = 15
        with self.assertRaisesRegex(ValueError, 'checkpoint_open'):
            self.check(pairs, typed)

    def test_tail_must_be_serial(self):
        pairs, typed = fixture()
        pairs[-2]['exit_sequence'] = 15
        with self.assertRaisesRegex(ValueError, 'overlapping_tail'):
            self.check(pairs, typed)

    def test_enter_order_must_match_actual_api_ids(self):
        pairs, typed = fixture()
        pairs[1]['enter_sequence'], pairs[1]['exit_sequence'] = 1, 16
        with self.assertRaisesRegex(ValueError, 'enter_order'):
            self.check(pairs, typed)

    def test_primary_release_before_library_unload_is_outside_profile(self):
        pairs, typed = fixture()
        pairs[-2]['symbol'], pairs[-1]['symbol'] = pairs[-1]['symbol'], pairs[-2]['symbol']
        typed[-2], typed[-1] = dict(typed[-1], api_id=5), dict(typed[-2], api_id=6)
        with self.assertRaisesRegex(ValueError, 'before_library_unloads'):
            self.check(pairs, typed)

    def test_strict_fields_reject_handles_and_asserted_closure(self):
        for field in ('address', 'native_handle', 'context_destroyed', 'observer_complete'):
            pairs, typed = fixture()
            typed[-1][field] = 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'fields'):
                self.check(pairs, typed)

    def test_boolean_negative_and_overflow_are_not_ids(self):
        for field in ('api_id', 'producer_id', 'context_id', 'device'):
            for value in (True, -1, 2**64):
                pairs, typed = fixture()
                typed[-1][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check(pairs, typed)

    def test_missing_or_duplicate_api_instances(self):
        for remove in (True, False):
            pairs, typed = fixture()
            if remove:
                del pairs[2]
            else:
                typed.append(copy.deepcopy(typed[-1]))
            with self.subTest(remove=remove), self.assertRaises(ValueError):
                self.check(pairs, typed)

    def test_no_tail_or_no_primary_retention(self):
        pairs, typed = fixture()
        with self.assertRaises(ValueError):
            self.check(pairs[:4], typed[:2])
        pairs, typed = fixture()
        pairs[0]['symbol'] = 'cuCtxGetCurrent'
        with self.assertRaises(ValueError):
            self.check(pairs, typed[1:])

    def test_snapshot_inputs_are_not_modified(self):
        inputs = fixture()
        before = copy.deepcopy(inputs)
        self.check(*inputs)
        self.assertEqual(inputs, before)


if __name__ == '__main__':
    unittest.main()
