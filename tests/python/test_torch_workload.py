from __future__ import annotations

import unittest

from xvram.torch_workload import (
    Llama2LikeConfig,
    acceptance_layers,
    build_llama2_like,
    deterministic_values,
)
from xvram.torch_planner import supported_operator_targets

try:
    import torch
except ImportError:
    torch = None


class TorchWorkloadTests(unittest.TestCase):
    def test_acceptance_byte_counts_are_frozen(self) -> None:
        expected = {
            16: 7_000_563_712,
            23: 9_833_930_752,
            31: 13_072_064_512,
        }
        for layers, _ratio in acceptance_layers():
            self.assertEqual(Llama2LikeConfig(layers=layers).parameter_bytes(), expected[layers])

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_chunked_generation_matches_whole(self) -> None:
        whole = deterministic_values(100, 19, seed=123, dtype=torch.float32)
        chunked = torch.cat(
            (
                deterministic_values(100, 7, seed=123, dtype=torch.float32),
                deterministic_values(107, 12, seed=123, dtype=torch.float32),
            )
        )
        torch.testing.assert_close(whole, chunked, rtol=0, atol=0)

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_tiny_model_executes(self) -> None:
        config = Llama2LikeConfig(
            layers=2,
            vocab_size=64,
            hidden=32,
            intermediate=48,
            heads=4,
            batch=1,
            sequence=8,
        )
        model = build_llama2_like(config, device="cpu", dtype=torch.float32)
        output = model(torch.arange(8).reshape(1, 8))
        self.assertEqual(tuple(output.shape), (1, 8, 64))

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_strict_export_stays_inside_phase4b_allowlist(self) -> None:
        config = Llama2LikeConfig(
            layers=1,
            vocab_size=64,
            hidden=32,
            intermediate=48,
            heads=4,
            batch=1,
            sequence=8,
        )
        model = build_llama2_like(config, device="meta", dtype=torch.float32).eval()
        exported = torch.export.export(model, (torch.arange(8).reshape(1, 8),), strict=True)
        targets = {
            str(node.target)
            for node in exported.graph_module.graph.nodes
            if node.op == "call_function"
        }
        self.assertLessEqual(targets, set(supported_operator_targets()))

    def test_invalid_head_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Llama2LikeConfig(hidden=30, heads=4)


if __name__ == "__main__":
    unittest.main()
