from __future__ import annotations

import unittest

from xvram.torch_acceptance import (
    DeterministicLlamaStateProvider,
    deterministic_token_ids,
    output_digest,
    output_error,
    prepare_meta_llama,
    run_layer_streamed_reference,
    strict_export_meta_llama,
)
from xvram.torch_workload import Llama2LikeConfig, build_llama2_like

try:
    import torch
except (ImportError, OSError):
    torch = None


def _tiny_config(*, layers: int = 1) -> Llama2LikeConfig:
    return Llama2LikeConfig(
        layers=layers,
        vocab_size=32,
        hidden=16,
        intermediate=24,
        heads=4,
        batch=1,
        sequence=4,
    )


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TorchAcceptanceTests(unittest.TestCase):
    def test_streamed_chunks_are_order_independent_and_byte_exact(self) -> None:
        prepared = prepare_meta_llama(
            _tiny_config(), seed=1234, dtype=torch.float32, chunk_bytes=17
        )
        provider = prepared.state_provider
        target = "layers.0.q_proj.weight"
        source = provider.source(target)

        raw_chunks = torch.cat([chunk.data for chunk in provider.iter_chunks(target)])
        materialized = provider.materialize_tensor(target)
        self.assertTrue(
            torch.equal(raw_chunks, materialized.contiguous().view(torch.uint8).reshape(-1))
        )
        first = provider.read_bytes(target, 3, 29).clone()
        provider.read_bytes(target, 101, 7)
        second = provider.read_bytes(target, 3, 29)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(source.nbytes(), materialized.numel() * materialized.element_size())
        self.assertEqual(provider.parameter_bytes, prepared.config.parameter_bytes(4))

    def test_tied_meta_weights_keep_one_storage_identity(self) -> None:
        config = _tiny_config()
        module = build_llama2_like(config, device="meta", dtype=torch.float32).eval()
        module.lm_head.weight = module.embedding.weight
        provider = DeterministicLlamaStateProvider(
            module, config, seed=9, dtype=torch.float32, chunk_bytes=31
        )

        embedding = provider.describe("embedding.weight")
        head = provider.describe("lm_head.weight")
        self.assertEqual(embedding.storage_key, head.storage_key)
        self.assertIs(
            provider.source("embedding.weight").storage_identity,
            provider.source("lm_head.weight").storage_identity,
        )
        self.assertTrue(
            torch.equal(
                provider.materialize_tensor("embedding.weight"),
                provider.materialize_tensor("lm_head.weight"),
            )
        )
        self.assertEqual(
            provider.parameter_bytes,
            config.parameter_bytes(4) - config.vocab_size * config.hidden * 4,
        )

    def test_input_digest_and_error_are_deterministic(self) -> None:
        config = _tiny_config()
        first = deterministic_token_ids(config, seed=77)
        second = deterministic_token_ids(config, seed=77)
        different = deterministic_token_ids(config, seed=78)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertEqual(first.dtype, torch.int64)
        self.assertEqual(tuple(first.shape), (1, 4))
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLess(int(first.max()), config.vocab_size)

        values = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        close = values + torch.tensor([[1.0e-5, -1.0e-5]])
        self.assertEqual(output_digest(values), output_digest(values.clone()))
        self.assertNotEqual(output_digest(values), output_digest(close))
        accepted = output_error(close, values, atol=1.0e-4, rtol=0.0)
        rejected = output_error(close, values, atol=1.0e-6, rtol=0.0)
        self.assertTrue(accepted.within_tolerance)
        self.assertFalse(rejected.within_tolerance)
        self.assertEqual(rejected.mismatch_count, 2)

    def test_strict_meta_export_uses_streaming_state(self) -> None:
        exported = strict_export_meta_llama(
            _tiny_config(), seed=5, dtype=torch.float32, chunk_bytes=23
        )
        self.assertTrue(exported.captured.plan.no_fallback)
        self.assertEqual(
            exported.captured.plan.logical_state_bytes,
            exported.prepared.state_provider.logical_state_bytes,
        )
        self.assertEqual(exported.captured.plan.input_names, ("token_ids",))
        self.assertEqual(tuple(exported.prepared.example_inputs[0].shape), (1, 4))

    def test_cpu_layer_streamed_reference_matches_full_model(self) -> None:
        config = _tiny_config(layers=2)
        prepared = prepare_meta_llama(
            config, seed=456, dtype=torch.float32, chunk_bytes=37
        )
        ordinary = build_llama2_like(config, device="cpu", dtype=torch.float32).eval()
        ordinary_state = dict(ordinary.named_parameters(remove_duplicate=False))
        ordinary_state.update(dict(ordinary.named_buffers(remove_duplicate=False)))
        with torch.no_grad():
            for target in prepared.state_provider.targets():
                ordinary_state[target].copy_(
                    prepared.state_provider.materialize_tensor(target)
                )
            expected = ordinary(prepared.example_inputs[0]).contiguous()

        progress = []
        result = run_layer_streamed_reference(
            prepared,
            device="cpu",
            sdpa_backend="math",
            progress=progress.append,
        )

        torch.testing.assert_close(result.output, expected, rtol=0, atol=0)
        self.assertEqual(result.digest, output_digest(expected))
        self.assertEqual(result.layers_completed, config.layers)
        self.assertGreater(result.peak_decoder_weight_bytes, 0)
        self.assertEqual(
            [item.stage for item in progress],
            ["embedding", "decoder", "decoder", "output"],
        )


if __name__ == "__main__":
    unittest.main()
