import copy
import unittest

from xvram.compat_audit_wddm_lifetime import analyze_lifetimes


def capture():
    return dict(start_ticks=1, end_ticks=100, reap_ticks=90, clock_frequency=10_000_000,
                events_lost=0, log_buffers_lost=0, realtime_buffers_lost=0, decoder_errors=0,
                etl_closed=True, trace_truncated=False, controller_reaped=True,
                process_tree_drained=True, timed_out=False, worker_exit_code=0)


def numbered(rows):
    for sequence, row in enumerate(rows, 1):
        row['sequence'] = sequence
    return rows


def fixture():
    process = dict(process_id=1, process_generation=1)
    device = dict(device_id=2, device_generation=1)
    context = dict(context_id=3, context_generation=1)
    queue = dict(queue_id=4, queue_generation=1)
    pc = dict(parent_context_id=0, parent_context_generation=0)
    pq = dict(parent_queue_id=0, parent_queue_generation=0)
    return numbered([
        dict(kind='process_start', timestamp_ticks=2, **process,
             parent_process_id=0, parent_process_generation=0),
        dict(kind='device_start', timestamp_ticks=3, **device, **process),
        dict(kind='context_start', timestamp_ticks=4, **context, **device, **pc),
        dict(kind='queue_start', timestamp_ticks=5, **queue, **context, **pq),
        dict(kind='queue_stop', timestamp_ticks=6, **queue, **context, **pq),
        dict(kind='context_stop', timestamp_ticks=7, **context, **device, **pc),
        dict(kind='device_stop', timestamp_ticks=8, **device, **process),
        dict(kind='process_stop', timestamp_ticks=9, **process, exit_code=0),
    ])


def parent_graph():
    rows = fixture()
    # A companion context and queue may have parents in another context of
    # the same process. Do not assume CUDA/WDDM or context/queue is one-to-one.
    child_context = dict(context_id=5, context_generation=1, device_id=2, device_generation=1,
                         parent_context_id=3, parent_context_generation=1)
    child_queue = dict(queue_id=6, queue_generation=1, context_id=5, context_generation=1,
                       parent_queue_id=4, parent_queue_generation=1)
    rows[4:4] = [dict(kind='context_start', timestamp_ticks=6, **child_context),
                 dict(kind='queue_start', timestamp_ticks=7, **child_queue),
                 dict(kind='queue_stop', timestamp_ticks=8, **child_queue),
                 dict(kind='context_stop', timestamp_ticks=9, **child_context)]
    for row, timestamp in zip(rows[8:], (10, 11, 12, 13)):
        row['timestamp_ticks'] = timestamp
    return numbered(rows)


class WddmLifetimeTests(unittest.TestCase):
    def check(self, rows=None, metadata=None):
        return analyze_lifetimes(fixture() if rows is None else rows,
                                 capture() if metadata is None else metadata,
                                 root_process=(1, 1))

    def test_balanced_observation_never_becomes_global_proof(self):
        result = self.check()
        self.assertEqual(result['counts']['events'], 8)
        self.assertTrue(result['capture_encloses_birth_and_reap'])
        self.assertTrue(result['observed_lifetimes_balanced'])
        self.assertFalse(result['successful_resource_destruction_proven'])
        self.assertFalse(result['observer_lifetime_proven'])
        self.assertTrue(all(value is False for value in result['proof'].values()))

    def test_all_loss_channels_fail_closed(self):
        for key in ('events_lost', 'log_buffers_lost', 'realtime_buffers_lost', 'decoder_errors'):
            for value in (1, -1, True, None):
                data = capture()
                data[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.check(metadata=data)

    def test_loss_channel_cannot_be_omitted(self):
        data = capture()
        del data['events_lost']
        with self.assertRaisesRegex(ValueError, 'fields'):
            self.check(metadata=data)

    def test_capture_must_start_before_birth_and_end_after_reap(self):
        for field, value in [('start_ticks', 2), ('end_ticks', 90), ('reap_ticks', 8),
                             ('clock_frequency', 0), ('etl_closed', False),
                             ('trace_truncated', True), ('controller_reaped', False),
                             ('process_tree_drained', False), ('timed_out', True),
                             ('worker_exit_code', 1)]:
            data = capture()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(metadata=data)

    def test_native_handles_and_false_semantic_assertions_rejected(self):
        for key in ('hContext', 'address', 'native_handle', 'successful_destruction'):
            rows = fixture()
            rows[2][key] = 123456
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'fields'):
                self.check(rows)

    def test_unknown_or_rundown_is_not_a_creation(self):
        for kind in ('context_dcstart', 'queue_packet', 'unknown'):
            rows = fixture()
            rows[2]['kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check(rows)

    def test_orphan_and_stale_generation_stops(self):
        for index, field in [(4, 'queue_id'), (4, 'queue_generation'),
                             (5, 'context_id'), (5, 'context_generation'),
                             (6, 'device_id'), (7, 'process_generation')]:
            rows = fixture()
            rows[index][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(rows)

    def test_resource_parent_must_exist(self):
        for index, field in [(2, 'parent_context'), (3, 'parent_queue')]:
            rows = fixture()
            rows[index][field + '_id'] = 33
            rows[index][field + '_generation'] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(rows)

    def test_partial_null_key_is_rejected(self):
        rows = fixture()
        rows[2]['parent_context_generation'] = 1
        with self.assertRaisesRegex(ValueError, 'partial_null'):
            self.check(rows)

    def test_wrong_parent_or_owner_at_stop(self):
        for index, field in [(4, 'context_id'), (5, 'device_id'), (6, 'process_id'),
                             (4, 'parent_queue_id'), (5, 'parent_context_id')]:
            rows = fixture()
            rows[index][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(rows)

    def test_context_stop_cannot_precede_queue_stop(self):
        rows = fixture()
        rows[4], rows[5] = rows[5], rows[4]
        rows[4]['timestamp_ticks'], rows[5]['timestamp_ticks'] = 6, 7
        with self.assertRaisesRegex(ValueError, 'live_children'):
            self.check(numbered(rows))

    def test_device_stop_cannot_precede_context_stop(self):
        rows = fixture()
        rows[5], rows[6] = rows[6], rows[5]
        rows[5]['timestamp_ticks'], rows[6]['timestamp_ticks'] = 7, 8
        with self.assertRaisesRegex(ValueError, 'live_contexts'):
            self.check(numbered(rows))

    def test_resource_cleanup_after_exit_and_reap_is_observed_not_assumed(self):
        rows = fixture()
        process_stop = rows.pop()
        process_stop['timestamp_ticks'] = 6
        rows.insert(4, process_stop)
        for row, timestamp in zip(rows[5:], (91, 92, 93)):
            row['timestamp_ticks'] = timestamp
        result = self.check(numbered(rows))
        self.assertEqual(result['counts']['resource_stops_after_process_exit'], 3)
        self.assertEqual(result['counts']['resource_stops_after_reap'], 3)
        self.assertFalse(result['successful_resource_destruction_proven'])

    def test_new_resource_after_owner_exit_is_rejected(self):
        rows = fixture()
        stop = rows.pop()
        stop['timestamp_ticks'] = 3
        rows.insert(1, stop)
        with self.assertRaises(ValueError):
            self.check(numbered(rows))

    def test_sequence_and_timestamps_are_strict(self):
        for field, value in [('sequence', 1), ('timestamp_ticks', 2), ('timestamp_ticks', True)]:
            rows = fixture()
            rows[3][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(rows)

    def test_missing_stop_is_not_filled_in_at_reap(self):
        rows = fixture()[:-1]
        with self.assertRaisesRegex(ValueError, 'incomplete_lifetime'):
            self.check(rows)

    def test_queue_generation_reuse_requires_completed_strict_boundary(self):
        rows = fixture()
        again_start = dict(rows[3], timestamp_ticks=7, queue_generation=2)
        again_stop = dict(rows[4], timestamp_ticks=8, queue_generation=2)
        rows[5]['timestamp_ticks'], rows[6]['timestamp_ticks'], rows[7]['timestamp_ticks'] = 9, 10, 11
        rows[5:5] = [again_start, again_stop]
        rows = numbered(rows)
        self.assertEqual(self.check(rows)['counts']['queue_generations'], 2)
        for generation, timestamp in [(1, 7), (3, 7), (2, 6)]:
            changed = copy.deepcopy(rows)
            changed[5].update(queue_generation=generation, timestamp_ticks=timestamp)
            with self.subTest(generation=generation, timestamp=timestamp), self.assertRaises(ValueError):
                self.check(changed)

    def test_child_process_parent_generation_and_pid_reuse(self):
        rows = fixture()
        rows[-1]['timestamp_ticks'] = 20
        additions = [dict(kind='process_start', timestamp_ticks=10, process_id=5, process_generation=1,
                          parent_process_id=1, parent_process_generation=1),
                     dict(kind='process_stop', timestamp_ticks=11, process_id=5, process_generation=1, exit_code=0),
                     dict(kind='process_start', timestamp_ticks=12, process_id=5, process_generation=2,
                          parent_process_id=1, parent_process_generation=1),
                     dict(kind='process_stop', timestamp_ticks=13, process_id=5, process_generation=2, exit_code=0)]
        rows[-1:-1] = additions
        rows = numbered(rows)
        self.assertEqual(self.check(rows)['counts']['process_generations'], 3)
        for index, field, value in [(9, 'process_generation', 1), (7, 'parent_process_generation', 2),
                                    (7, 'parent_process_id', 99)]:
            bad = copy.deepcopy(rows)
            bad[index][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(bad)

    def test_companion_context_and_cross_context_parent_queue(self):
        result = self.check(parent_graph())
        self.assertEqual(result['counts']['context_generations'], 2)
        self.assertEqual(result['counts']['queue_generations'], 2)

    def test_parent_queue_cannot_stop_while_child_queue_lives(self):
        rows = parent_graph()
        rows[6], rows[8] = rows[8], rows[6]
        rows[6]['timestamp_ticks'], rows[8]['timestamp_ticks'] = 8, 10
        with self.assertRaisesRegex(ValueError, 'queue_stop_with_live_children'):
            self.check(numbered(rows))

    def test_parent_context_cannot_stop_while_companion_context_lives(self):
        rows = parent_graph()
        # Remove all queues so the context-child condition, rather than queue
        # ownership, is the specific reason to reject this reversed order.
        rows = [row for row in rows if not row['kind'].startswith('queue_')]
        child_stop = next(i for i, row in enumerate(rows)
                          if row['kind'] == 'context_stop' and row['context_id'] == 5)
        parent_stop = next(i for i, row in enumerate(rows)
                           if row['kind'] == 'context_stop' and row['context_id'] == 3)
        rows[child_stop], rows[parent_stop] = rows[parent_stop], rows[child_stop]
        rows[child_stop]['timestamp_ticks'], rows[parent_stop]['timestamp_ticks'] = 9, 11
        with self.assertRaisesRegex(ValueError, 'context_stop_with_live_children'):
            self.check(numbered(rows))

    def test_parent_relationship_cannot_cross_process_ownership(self):
        for parent_kind in ('context', 'queue'):
            rows = fixture()[:4]
            rows += [dict(kind='process_start', timestamp_ticks=6, process_id=7, process_generation=1,
                          parent_process_id=1, parent_process_generation=1),
                     dict(kind='device_start', timestamp_ticks=7, process_id=7, process_generation=1,
                          device_id=8, device_generation=1),
                     dict(kind='context_start', timestamp_ticks=8, device_id=8, device_generation=1,
                          context_id=9, context_generation=1,
                          parent_context_id=3 if parent_kind == 'context' else 0,
                          parent_context_generation=1 if parent_kind == 'context' else 0)]
            if parent_kind == 'queue':
                rows.append(dict(kind='queue_start', timestamp_ticks=9, context_id=9, context_generation=1,
                                 queue_id=10, queue_generation=1, parent_queue_id=4, parent_queue_generation=1))
            with self.subTest(parent_kind=parent_kind), self.assertRaisesRegex(ValueError, 'parent_mismatch'):
                self.check(numbered(rows))

    def test_live_id_cannot_be_reused_with_new_generation(self):
        rows = fixture()
        rows.insert(4, dict(rows[3], timestamp_ticks=6, queue_generation=2))
        for row, timestamp in zip(rows[5:], (7, 8, 9, 10)):
            row['timestamp_ticks'] = timestamp
        with self.assertRaisesRegex(ValueError, 'reuse_live_resource'):
            self.check(numbered(rows))

    def test_failed_process_stop_and_input_overflow(self):
        for value in (1, -1, True, 2**64):
            rows = fixture()
            rows[-1]['exit_code'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(rows)

    def test_child_can_outlive_parent_but_must_stop_before_reap(self):
        rows = fixture()
        rows.insert(7, dict(kind='process_start', timestamp_ticks=9, process_id=5, process_generation=1,
                           parent_process_id=1, parent_process_generation=1))
        rows[-1]['timestamp_ticks'] = 10
        rows.append(dict(kind='process_stop', timestamp_ticks=11, process_id=5, process_generation=1,
                         exit_code=0))
        self.assertEqual(self.check(numbered(rows))['counts']['process_generations'], 2)

    def test_empty_invalid_root_and_unknown_fields(self):
        for root in ((1, 2), (True, 1), (0, 0), (1,), '1'):
            with self.subTest(root=root), self.assertRaises(ValueError):
                analyze_lifetimes(fixture(), capture(), root_process=root)
        with self.assertRaises(ValueError):
            self.check([])

    def test_inputs_remain_unchanged(self):
        rows, metadata = fixture(), capture()
        original = copy.deepcopy((rows, metadata))
        self.check(rows, metadata)
        self.assertEqual((rows, metadata), original)


if __name__ == '__main__':
    unittest.main()
