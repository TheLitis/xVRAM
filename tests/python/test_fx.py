from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from xvram.fx import analyze_next_uses  # noqa: E402


class _Node:
    def __init__(self, name, op, inputs=()):
        self.name = name
        self.op = op
        self.all_input_nodes = tuple(inputs)


class _Graph:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)


def _diamond_graph():
    input_node = _Node("input_1", "placeholder")
    weight = _Node("weight", "get_attr")
    shared = _Node("shared", "call_function", (input_node, weight))
    left = _Node("left", "call_function", (shared,))
    right = _Node("right", "call_function", (shared,))
    output = _Node("output", "output", (left, right))
    return _Graph((input_node, weight, shared, left, right, output))


class FxAnalysisTests(unittest.TestCase):
    def test_next_use_and_last_use_are_deterministic(self):
        plan = analyze_next_uses(_diamond_graph(), prefetch_distance=2)

        shared = plan.value("shared")
        self.assertEqual(shared.use_indices, (3, 4))
        self.assertEqual(shared.next_use_after(2), 3)
        self.assertEqual(shared.next_use_after(3), 4)
        self.assertIsNone(shared.next_use_after(4))

        left_read = plan.steps[3].reads[0]
        right_read = plan.steps[4].reads[0]
        self.assertEqual(left_read.next_use_index, 4)
        self.assertIsNone(right_read.next_use_index)
        self.assertEqual(plan.steps[3].release_candidates, ())
        self.assertEqual(plan.steps[4].release_candidates, ("shared",))

    def test_external_persistent_and_returned_values_are_not_release_candidates(self):
        plan = analyze_next_uses(_diamond_graph())

        all_candidates = {
            name for step in plan.steps for name in step.release_candidates
        }
        self.assertNotIn("input_1", all_candidates)
        self.assertNotIn("weight", all_candidates)
        self.assertNotIn("left", all_candidates)
        self.assertNotIn("right", all_candidates)
        self.assertTrue(plan.value("left").escapes_graph)
        self.assertTrue(plan.advisory_only)

    def test_prefetch_window_is_bounded_and_zero_disables_it(self):
        disabled = analyze_next_uses(_diamond_graph(), prefetch_distance=0)
        enabled = analyze_next_uses(_diamond_graph(), prefetch_distance=2)

        self.assertTrue(all(not step.prefetch_candidates for step in disabled.steps))
        self.assertNotIn("input_1", enabled.steps[0].prefetch_candidates)
        self.assertIn("input_1", enabled.steps[1].prefetch_candidates)

    def test_steps_are_post_submission_and_dead_results_are_releasable(self):
        input_node = _Node("input_1", "placeholder")
        produced_now = _Node("produced_now", "call_function", (input_node,))
        dead = _Node("dead", "call_function", (input_node,))
        future = _Node("future", "call_function", (produced_now,))
        output = _Node("output", "output", (future,))
        plan = analyze_next_uses(
            _Graph((input_node, produced_now, dead, future, output)),
            prefetch_distance=2,
        )

        self.assertEqual(plan.steps[2].release_candidates, ("dead",))
        self.assertNotIn("produced_now", plan.steps[1].prefetch_candidates)
        self.assertIn("produced_now", plan.steps[2].prefetch_candidates)

    def test_invalid_prefetch_distance_is_rejected(self):
        for value in (-1, 9):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    analyze_next_uses(_diamond_graph(), prefetch_distance=value)
        with self.assertRaises(TypeError):
            analyze_next_uses(_diamond_graph(), prefetch_distance=True)

    def test_non_topological_graph_is_rejected(self):
        future = _Node("future", "call_function")
        early = _Node("early", "call_function", (future,))
        graph = _Graph((early, future))
        with self.assertRaisesRegex(ValueError, "topological"):
            analyze_next_uses(graph)


if __name__ == "__main__":
    unittest.main()
