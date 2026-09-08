"""Synthetic witness facts test implications, not actual application coverage."""
from dataclasses import replace
import itertools
import unittest
from unittest import mock

from xvram.compat_audit_proof import DeviceOrder, TensorRange, memory_envelopes


class MemoryProofTests(unittest.TestCase):
    def setUp(self):
        self.span = TensorRange((1, 1), 256, 64, 64, 0, 64, "read")

    def check(self, *ranges):
        return memory_envelopes(list(ranges), chunk_bytes=64)

    def test_tensor_bounds_are_stricter_than_allocation_bounds(self):
        self.assertEqual(self.check(self.span)["rounded_bytes_upper_bound"], 64)
        with self.assertRaisesRegex(ValueError, "outside_tensor"):
            self.check(replace(self.span, access_bytes=65))

    def test_tensor_itself_cannot_exceed_allocation(self):
        with self.assertRaisesRegex(ValueError, "tensor_outside"):
            self.check(replace(self.span, tensor_offset=240))

    def test_partial_tail_and_nonzero_base_alignment(self):
        value = self.check(replace(self.span, base_chunk_offset=1))
        self.assertEqual(value["rounded_bytes_upper_bound"], 128)
        self.assertFalse(value["admission_ready"])

    def test_overlapping_reads_allowed_write_alias_forbidden(self):
        self.assertEqual(self.check(self.span, self.span)["chunk_count_upper_bound"], 1)
        for mode in ("write", "read_write"):
            with self.assertRaisesRegex(ValueError, "overlapping_write"):
                self.check(self.span, replace(self.span, mode=mode))

    def test_adjacent_writes_allowed_empty_writes_do_not_alias(self):
        first = replace(self.span, access_bytes=32, mode="write")
        second = replace(first, access_offset=32)
        self.check(first, second, replace(first, access_bytes=0))

    def test_read_envelope_holes_conservatively_reject_write(self):
        with self.assertRaises(ValueError):
            self.check(self.span, replace(self.span, access_offset=20, access_bytes=1, mode="write"))

    def test_generation_and_layout_inconsistency(self):
        for other in (replace(self.span, allocation=(1, 2)), replace(self.span, allocation_bytes=512)):
            with self.assertRaises(ValueError):
                self.check(self.span, other)

    def test_integer_overflow_bool_and_invalid_ranges(self):
        for field in ("allocation_bytes", "tensor_offset", "tensor_bytes", "access_offset", "access_bytes"):
            for value in (-1, True, 1 << 64):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.check(replace(self.span, **{field: value}))
        with self.assertRaises(ValueError):
            self.check(replace(self.span, allocation=(True, 1)))
        with self.assertRaises(ValueError):
            self.check(replace(self.span, mode="unknown"))

    def test_small_alias_cases_against_byte_sets(self):
        for a, b, alen, blen, awrite, bwrite in itertools.product(range(4), range(4), range(3), range(3), (False, True), (False, True)):
            x = replace(self.span, access_offset=a, access_bytes=alen, mode="write" if awrite else "read")
            y = replace(self.span, access_offset=b, access_bytes=blen, mode="write" if bwrite else "read")
            conflict = bool(set(range(a, a+alen)) & set(range(b, b+blen))) and (awrite or bwrite)
            if conflict:
                with self.assertRaises(ValueError):
                    self.check(x, y)
            else:
                self.check(x, y)


class OrderingProofTests(unittest.TestCase):
    def setUp(self):
        self.order = DeviceOrder()
        self.a, self.b = (1, 1), (2, 1)
        self.event = (1, 1)
        self.order.create_stream(self.a); self.order.create_stream(self.b)
        self.order.create_event(self.event)

    def test_host_submission_never_proves_gpu_completion(self):
        node = self.order.submit(self.a)
        self.assertFalse(self.order.retired(node))
        with self.assertRaisesRegex(ValueError, "early_reuse"):
            self.order.require_retired([node])

    def test_same_stream_order_not_cross_stream_order(self):
        x, y, z = self.order.submit(self.a), self.order.submit(self.a), self.order.submit(self.b)
        self.assertTrue(self.order.precedes(x, y))
        self.assertFalse(self.order.precedes(x, z))
        self.assertFalse(self.order.precedes(z, y))

    def test_wait_event_transitive_edges_and_not_ready(self):
        x = self.order.submit(self.a)
        ready = self.order.record(self.event, self.a)
        self.order.wait(self.b, ready)
        y = self.order.submit(self.b)
        done = self.order.record(self.event, self.b)
        self.assertTrue(self.order.precedes(x, y))
        self.order.query(done, "not_ready")
        self.assertFalse(self.order.retired(x))
        self.order.query(done, "success")
        self.order.require_retired([x, y])

    def test_rerecord_cannot_retroactively_extend_wait(self):
        first = self.order.record(self.event, self.a)
        self.order.wait(self.b, first)
        later = self.order.submit(self.a)
        second = self.order.record(self.event, self.a)
        bnode = self.order.submit(self.b)
        self.assertFalse(self.order.precedes(later, bnode))
        with self.assertRaises(ValueError):
            self.order.query(first, "success")
        self.order.query(second, "success")
        self.assertFalse(self.order.retired(bnode))

    def test_wait_uses_current_record_at_submission(self):
        old = self.order.record(self.event, self.a)
        self.order.record(self.event, self.a)
        with self.assertRaises(ValueError):
            self.order.wait(self.b, old)

    def test_event_aba_and_wait_snapshot_survives_destroy(self):
        node = self.order.submit(self.a)
        old = self.order.record(self.event, self.a)
        self.order.wait(self.b, old)
        self.order.destroy_event(self.event)
        self.order.create_event((1, 2))
        with self.assertRaises(ValueError):
            self.order.query(old, "success")
        done = self.order.record((1, 2), self.b)
        self.order.query(done, "success")
        self.assertTrue(self.order.retired(node))

    def test_unknown_query_poison_prevents_retirement_claim(self):
        token = self.order.record(self.event, self.a)
        with self.assertRaises(ValueError):
            self.order.query(token, "driver_error")
        self.assertFalse(self.order.retired(1))
        with self.assertRaises(ValueError):
            self.order.submit(self.a)

    def test_stream_generation_does_not_inherit_completion(self):
        node = self.order.submit(self.a)
        with self.assertRaises(ValueError):
            self.order.destroy_stream(self.a)
        token = self.order.record(self.event, self.a)
        self.order.query(token, "success")
        self.order.destroy_stream(self.a)
        self.order.create_stream((1, 2))
        other = self.order.submit((1, 2))
        self.assertTrue(self.order.retired(node))
        self.assertFalse(self.order.retired(other))
        with self.assertRaises(ValueError):
            self.order.submit(self.a)

    def test_all_frame_users_must_retire(self):
        x, y = self.order.submit(self.a), self.order.submit(self.b)
        token = self.order.record(self.event, self.a)
        self.order.query(token, "success")
        with self.assertRaises(ValueError):
            self.order.require_retired([x, y])
        with self.assertRaises(ValueError):
            self.order.require_retired([])

    def test_strict_id_and_capacity_checks(self):
        for stream in ((True, 1), (1, 0), (3, 2), (1, 1)):
            with self.assertRaises(ValueError):
                self.order.create_stream(stream)
        self.order.clock_entries = 1_000_000
        with self.assertRaisesRegex(ValueError, "clock_limit"):
            self.order.submit(self.a)
        self.assertFalse(self.order.nodes)
        with mock.patch("xvram.compat_audit_proof.LIMIT", 1), self.assertRaises(ValueError):
            self.order.create_event((2, 1))

    def test_terminal_requires_every_fact_and_every_retirement(self):
        token = self.order.record(self.event, self.a)
        facts = dict(producer_barrier=True, api_pairs_closed=True, flush_completed=True,
                     buffers_returned=True, records_consumed=True, dropped=0, errors=0)
        self.assertFalse(self.order.terminal(**facts))
        self.order.query(token, "success")
        self.assertFalse(self.order.terminal(**facts))
        self.order.seal_submissions()
        self.assertTrue(self.order.terminal(**facts))  # conditional synthetic model ONLY
        for key in facts:
            bad = {**facts, key: 1 if key in ("dropped", "errors") else False}
            self.assertFalse(self.order.terminal(**bad))
        with self.assertRaises(ValueError):
            self.order.submit(self.a)
        self.assertFalse(self.order.terminal(**facts))

    def test_barrier_blocks_new_work_but_allows_completion_drain(self):
        token = self.order.record(self.event, self.a)
        self.order.seal_submissions()
        self.order.query(token, "not_ready")
        self.assertFalse(self.order.retired(1))
        self.order.query(token, "success")
        self.assertTrue(self.order.retired(1))
        self.order.destroy_event(self.event)
        self.order.destroy_stream(self.a)

    def test_wait_capacity_failure_does_not_add_edges(self):
        token = self.order.record(self.event, self.a)
        self.order.clock_entries = 1_000_000
        with self.assertRaises(ValueError):
            self.order.wait(self.b, token)
        self.assertEqual(self.order.streams[self.b], {})

    def test_random_stream_dag_against_explicit_ancestors(self):
        import random
        rng = random.Random(713)
        ancestors, tails, recorded = {}, {}, None
        for _ in range(100):
            stream = rng.choice((self.a, self.b))
            prior = set(ancestors.get(tails.get(stream), set()))
            action = rng.randrange(3)
            if action == 0 and recorded:
                prior |= ancestors[recorded[1]]
                node = self.order.wait(stream, recorded[0])
            elif action == 1:
                token = self.order.record(self.event, stream)
                node = len(self.order.nodes)
                recorded = token, node
            else:
                node = self.order.submit(stream)
            ancestors[node] = prior | {node}
            tails[stream] = node
            for before in range(1, node+1):
                self.assertEqual(self.order.precedes(before, node), before in ancestors[node])


if __name__ == "__main__":
    unittest.main()
