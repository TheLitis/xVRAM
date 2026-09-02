from __future__ import annotations

import os
import gc
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from xvram.torch_allocator import XVRAM_TORCH_ALLOCATOR_ENV, load  # noqa: E402


class TorchAllocatorIntegrationTests(unittest.TestCase):
    def _torch_and_library(self):
        library = os.environ.get(XVRAM_TORCH_ALLOCATOR_ENV)
        if not library:
            self.skipTest("{} is not set".format(XVRAM_TORCH_ALLOCATOR_ENV))

        try:
            import torch
        except (ImportError, OSError) as error:
            self.skipTest("PyTorch is unavailable: {}".format(error))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable in PyTorch")
        return torch, library

    def test_cuda_mem_pool_callbacks_and_mapping_invariants(self):
        torch, library = self._torch_and_library()

        owner = load(library)
        pool = owner.create_mem_pool()
        pool.reset_stats()
        with pool:
            tensor = torch.arange(4096, dtype=torch.float32, device="cuda")
            result = tensor.square().sum()
            self.assertGreater(float(result.cpu()), 0.0)
            torch.cuda.synchronize()

        del result
        del tensor
        torch.cuda.synchronize()
        del pool
        gc.collect()
        torch.cuda.synchronize()

        stats = owner.get_stats()
        self.assertGreaterEqual(stats.allocation_calls, 1)
        self.assertEqual(stats.allocation_failures, 0)
        self.assertEqual(stats.maps, stats.set_access_calls)
        self.assertEqual(stats.maps, stats.unmaps)
        self.assertEqual(stats.handles_created, stats.handle_releases)
        self.assertEqual(stats.reservations, stats.reservation_frees)
        self.assertEqual(stats.free_calls, stats.event_boundaries)
        self.assertEqual(stats.active_segments, 0)
        self.assertEqual(stats.mapped_bytes_current, 0)
        self.assertEqual(stats.unsafe_unmaps, 0)
        self.assertEqual(stats.quarantined_segments, 0)

    def test_cross_stream_record_stream_retires_before_segment_free(self):
        torch, library = self._torch_and_library()
        owner = load(library)
        owner.reset_stats()
        pool = owner.create_mem_pool()
        consumer = torch.cuda.Stream()

        with pool:
            source = torch.arange(1024 * 1024, dtype=torch.float32, device="cuda")
            with torch.cuda.stream(consumer):
                output = source.square()
            source.record_stream(consumer)
            del source
            consumer.synchronize()
            self.assertGreater(float(output.sum().cpu()), 0.0)

        del output
        del consumer
        torch.cuda.synchronize()
        del pool
        gc.collect()
        torch.cuda.synchronize()

        stats = owner.get_stats()
        self.assertGreaterEqual(stats.allocation_calls, 1)
        self.assertEqual(stats.allocation_failures, 0)
        self.assertEqual(stats.free_failures, 0)
        self.assertEqual(stats.context_mismatches, 0)
        self.assertEqual(stats.stream_mismatches, 0)
        self.assertEqual(stats.maps, stats.unmaps)
        self.assertEqual(stats.free_calls, stats.event_boundaries)
        self.assertEqual(stats.quarantined_segments, 0)


if __name__ == "__main__":
    unittest.main()
