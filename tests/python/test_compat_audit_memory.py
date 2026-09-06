"""No-driver checks of declared source models; fixtures are not captured arguments."""

import itertools
import json
import random
import tempfile
import unittest
from unittest import mock

from xvram import compat_audit_memory as memory


def quantize_args(**changes):
    result = {
        "valid_columns": 35, "padded_columns": 64, "rows": 2, "channels": 1, "samples": 1,
        "stride_row_elements": 40, "stride_channel_elements": 80, "stride_sample_elements": 80,
        "grid": (2, 2, 1), "block": (32, 1, 1), "compiled_max_threads_per_block": 256,
    }
    result.update(changes)
    return result


def fixup_args(**changes):
    result = {
        "rows": 96, "columns_max": 16, "channels": 1, "samples": 1, "columns_x": 256,
        "tile_rows": 32, "tile_columns": 8, "sm_count": 4,
        "declared_tiles": 6, "declared_blocks": 4, "quant_type": 12,
        "stream_k_enabled": True, "nvidia_profile": True,
    }
    result.update(changes)
    return result


class QuantizeSourceRuleTests(unittest.TestCase):
    def test_tail_reads_only_valid_f32_and_writes_padded_blocks(self):
        result = memory._quantize_q8_1(**quantize_args())
        self.assertEqual(result["read_envelopes"][0]["length_bytes"], (40 + 35) * 4)
        self.assertEqual(result["write_extents"][0]["length_bytes"], 2 * 64 // 32 * 36)
        self.assertEqual(result["global_scratch_bytes"], 0)
        self.assertFalse(result["runtime_binding_proven"])
        self.assertFalse(result["admission_ready"])

    def test_multi_channel_sample_envelope(self):
        result = memory._quantize_q8_1(**quantize_args(
            channels=3, samples=2, stride_channel_elements=100,
            stride_sample_elements=400, grid=(2, 2, 6)))
        self.assertEqual(result["read_envelopes"][0]["length_bytes"], (400 + 200 + 40 + 35) * 4)
        self.assertEqual(result["write_extents"][0]["length_bytes"], 64 // 32 * 36 * 2 * 3 * 2)

    def test_broadcast_read_aliases_do_not_multiply_envelope(self):
        result = memory._quantize_q8_1(**quantize_args(
            rows=9, stride_row_elements=0, grid=(2, 9, 1)))
        self.assertEqual(result["read_envelopes"][0]["length_bytes"], 35 * 4)
        self.assertEqual(result["row_count"], 9)

    def test_padding_whole_blocks_and_partial_block_grid(self):
        result = memory._quantize_q8_1(**quantize_args(
            valid_columns=1, padded_columns=32, block=(256, 1, 1), grid=(1, 2, 1)))
        self.assertEqual(result["write_extents"][0]["length_bytes"], 72)
        self.assertEqual(result["read_envelopes"][0]["length_bytes"], 41 * 4)

    def test_small_shapes_against_bruteforce_source_indices(self):
        for valid, rows, channels, samples in itertools.product((1, 31, 32, 35), (1, 2), (1, 3), (1, 2)):
            padded = (valid + 31) // 32 * 32
            strides = (valid + 5, rows * (valid + 7), channels * rows * (valid + 11))
            result = memory._quantize_q8_1(**quantize_args(
                valid_columns=valid, padded_columns=padded, rows=rows, channels=channels, samples=samples,
                stride_row_elements=strides[0], stride_channel_elements=strides[1], stride_sample_elements=strides[2],
                grid=(padded // 32, rows, channels * samples)))
            reads = {s * strides[2] + c * strides[1] + r * strides[0] + column
                     for s in range(samples) for c in range(channels) for r in range(rows) for column in range(valid)}
            self.assertEqual(result["read_envelopes"][0]["length_bytes"], (max(reads) + 1) * 4)
            output_blocks = {(((s * channels + c) * rows + r) * padded + column) // 32
                             for s in range(samples) for c in range(channels) for r in range(rows)
                             for column in range(padded)}
            self.assertEqual(result["write_extents"][0]["length_bytes"], len(output_blocks) * 36)

    def test_invalid_dimensions_and_strides(self):
        for changes in ({"valid_columns": 0}, {"valid_columns": 65}, {"padded_columns": 33},
                        {"rows": 0}, {"channels": -1}, {"samples": True},
                        {"stride_row_elements": -1}, {"stride_sample_elements": 1.0}):
            with self.subTest(changes=changes), self.assertRaises(memory.MemoryRuleError):
                memory._quantize_q8_1(**quantize_args(**changes))

    def test_inconsistent_launch_geometry_and_thread_limit(self):
        for changes in ({"grid": (1, 2, 1)}, {"grid": (2, 1, 1)}, {"grid": (2, 2, 2)},
                        {"block": (31, 1, 1)}, {"block": (32, 2, 1)},
                        {"compiled_max_threads_per_block": 31}, {"block": (1025, 1, 1)},
                        {"grid": (1, 2)}, {"grid": (True, 2, 1)}):
            with self.subTest(changes=changes), self.assertRaises(memory.MemoryRuleError):
                memory._quantize_q8_1(**quantize_args(**changes))

    def test_outside_bounded_grid_profile(self):
        with self.assertRaises(memory.MemoryRuleError):
            memory._quantize_q8_1(**quantize_args(channels=256, samples=256, grid=(2, 2, 65536)))

    def test_signed_source_index_overflow_rejected(self):
        with self.assertRaisesRegex(memory.MemoryRuleError, "arithmetic_overflow"):
            memory._quantize_q8_1(**quantize_args(stride_row_elements=memory.I64_MAX))

    def test_byte_extent_overflow_rejected(self):
        with self.assertRaisesRegex(memory.MemoryRuleError, "arithmetic_overflow"):
            memory._quantize_q8_1(**quantize_args(stride_row_elements=memory.U64_MAX // 4))


class StreamKSourceRuleTests(unittest.TestCase):
    def test_global_fixup_payload_and_native_pool_reservation(self):
        result = memory._stream_k_fixup(**fixup_args())
        self.assertTrue(result["fixup_needed"])
        self.assertEqual(result["scratch_payload_bytes"], 4 * 4 * 32 * 8)
        self.assertEqual(result["native_vmm_suballocation_bytes"], 4096)
        self.assertIsNone(result["native_vmm_physical_growth_bytes"])
        self.assertFalse(result["admission_ready"])

    def test_high_efficiency_selects_tiling_without_fixup(self):
        result = memory._stream_k_fixup(**fixup_args(rows=128, declared_tiles=8, declared_blocks=8))
        self.assertEqual(result["tile_efficiency_percent"], 100)
        self.assertFalse(result["fixup_needed"])
        self.assertEqual(result["scratch_payload_bytes"], 0)
        self.assertEqual(result["native_vmm_suballocation_bytes"], 0)

    def test_ninety_percent_threshold(self):
        result = memory._stream_k_fixup(**fixup_args(
            rows=288, columns_max=8, sm_count=10, declared_tiles=9, declared_blocks=9))
        self.assertEqual(result["tile_efficiency_percent"], 90)
        self.assertFalse(result["fixup_needed"])
        below = memory._stream_k_fixup(**fixup_args(
            rows=256, columns_max=8, sm_count=10, declared_tiles=8, declared_blocks=10))
        self.assertEqual(below["tile_efficiency_percent"], 80)
        self.assertTrue(below["fixup_needed"])

    def test_small_shapes_match_bruteforce_tile_set(self):
        for rows, columns, channels, samples, sms in itertools.product((1, 32, 33, 64, 95), (1, 8, 9), (1, 2), (1, 2), (2, 3, 7)):
            tile_set = {(r // 32, c // 8, ch, sample)
                        for r in range(rows) for c in range(columns) for ch in range(channels) for sample in range(samples)}
            tiles = len(tile_set)
            waves = (tiles + sms - 1) // sms
            efficiency = 100 * tiles // (sms * waves)
            blocks = tiles if efficiency >= 90 else sms
            result = memory._stream_k_fixup(**fixup_args(
                rows=rows, columns_max=columns, channels=channels, samples=samples,
                sm_count=sms, declared_tiles=tiles, declared_blocks=blocks))
            expected = 4 * blocks * 32 * 8 if tiles % blocks else 0
            self.assertEqual(result["scratch_payload_bytes"], expected)

    def test_q6_k_has_same_fixup_geometry_rule(self):
        first = memory._stream_k_fixup(**fixup_args())
        second = memory._stream_k_fixup(**fixup_args(quant_type=14))
        self.assertEqual(first["scratch_payload_bytes"], second["scratch_payload_bytes"])

    def test_declared_configuration_must_reconcile(self):
        for changes in ({"declared_tiles": 5}, {"declared_blocks": 5}, {"tile_rows": 31},
                        {"tile_columns": 9}, {"quant_type": 13}, {"columns_x": 255},
                        {"stream_k_enabled": False}, {"nvidia_profile": False}, {"nvidia_profile": 1}):
            with self.subTest(changes=changes), self.assertRaises(memory.MemoryRuleError):
                memory._stream_k_fixup(**fixup_args(**changes))

    def test_source_signed_intermediates_cannot_overflow(self):
        with self.assertRaisesRegex(memory.MemoryRuleError, "arithmetic_overflow"):
            memory._stream_k_fixup(**fixup_args(rows=32 * 30_000_000, columns_max=8,
                                              declared_tiles=30_000_000, declared_blocks=30_000_000))

    def test_safe_python_ceil_cannot_admit_out_of_profile_numerators(self):
        cases = (
            {"rows": memory.I32_MAX, "tile_rows": 4096, "columns_max": 8,
             "declared_tiles": 524288, "declared_blocks": 524288},
            {"rows": 32, "columns_max": memory.I32_MAX, "tile_columns": 128,
             "declared_tiles": 16777216, "declared_blocks": 16777216},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(memory.MemoryRuleError, "arithmetic_overflow"):
                memory._stream_k_fixup(**fixup_args(**changes))

    def test_ceil_numerator_exact_signed_boundary_is_accepted(self):
        cases = (
            {"rows": memory.I32_MAX - 4095, "tile_rows": 4096, "columns_max": 8,
             "declared_tiles": 524287, "declared_blocks": 524287},
            {"rows": 32, "columns_max": memory.I32_MAX - 127, "tile_columns": 128,
             "declared_tiles": 16777215, "declared_blocks": 16777215},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                result = memory._stream_k_fixup(**fixup_args(**changes))
                self.assertEqual(result["tile_count"], changes["declared_tiles"])
                self.assertFalse(result["fixup_needed"])

    def test_linear_k_index_limit(self):
        with self.assertRaisesRegex(memory.MemoryRuleError, "stream_k_linear_index_limit"):
            memory._stream_k_fixup(**fixup_args(rows=32 * 1024, columns_max=8,
                                              columns_x=256 * (1 << 20), declared_tiles=1024, declared_blocks=1024))


class ChunkUnionTests(unittest.TestCase):
    def r(self, offset=0, length=1, *, allocation=1, generation=1, size=1000, base=0):
        return memory.AllocationRange(allocation, generation, size, base, offset, length)

    def test_unaligned_base_adds_extra_chunk(self):
        result = memory.chunk_rounded_union([self.r(length=16, base=1)], chunk_bytes=16)
        self.assertEqual(result["chunk_count_upper_bound"], 2)
        self.assertEqual(result["rounded_bytes_upper_bound"], 32)
        self.assertEqual(result["allocations"][0]["relative_chunk_intervals"], [[0, 2]])

    def test_overlapping_and_adjacent_ranges_merge(self):
        result = memory.chunk_rounded_union([self.r(0, 8), self.r(4, 13), self.r(17, 15)], chunk_bytes=16)
        self.assertEqual(result["chunk_count_upper_bound"], 2)
        self.assertEqual(result["allocations"][0]["relative_chunk_intervals"], [[0, 2]])

    def test_disjoint_ranges_keep_gap(self):
        result = memory.chunk_rounded_union([self.r(0, 1), self.r(48, 1)], chunk_bytes=16)
        self.assertEqual(result["allocations"][0]["relative_chunk_intervals"], [[0, 1], [3, 4]])

    def test_distinct_allocations_do_not_assume_shared_physical_chunks(self):
        result = memory.chunk_rounded_union([self.r(), self.r(allocation=2)], chunk_bytes=16)
        self.assertEqual(result["chunk_count_upper_bound"], 2)
        self.assertIsNone(result["exact_physical_residency_bytes"])

    def test_zero_length_range_at_end_touches_nothing(self):
        result = memory.chunk_rounded_union([self.r(1000, 0)], chunk_bytes=16)
        self.assertEqual(result["chunk_count_upper_bound"], 0)

    def test_stale_generation_ambiguity_rejected(self):
        with self.assertRaisesRegex(memory.MemoryRuleError, "ambiguous_allocation_generation"):
            memory.chunk_rounded_union([self.r(), self.r(generation=2)], chunk_bytes=16)

    def test_layout_contradictions_rejected(self):
        for other in (self.r(base=1), self.r(size=1001)):
            with self.assertRaisesRegex(memory.MemoryRuleError, "inconsistent_allocation_layout"):
                memory.chunk_rounded_union([self.r(), other], chunk_bytes=16)

    def test_ranges_and_base_must_be_explicit_and_valid(self):
        for item in (self.r(offset=-1), self.r(length=-1), self.r(base=16), self.r(generation=0),
                     self.r(offset=999, length=2), self.r(allocation=True)):
            with self.subTest(item=item), self.assertRaises(memory.MemoryRuleError):
                memory.chunk_rounded_union([item], chunk_bytes=16)
        with self.assertRaises(TypeError):
            memory.AllocationRange(allocation_id=1, generation=1, allocation_bytes=100, offset_bytes=0, length_bytes=1)

    def test_overflow_in_offsets_or_rounded_total_rejected(self):
        for item in (self.r(offset=memory.U64_MAX, length=1, size=memory.U64_MAX),
                     self.r(offset=memory.U64_MAX - 1, length=1, size=memory.U64_MAX, base=1)):
            with self.assertRaisesRegex(memory.MemoryRuleError, "arithmetic_overflow"):
                memory.chunk_rounded_union([item], chunk_bytes=16)
        with self.assertRaisesRegex(memory.MemoryRuleError, "arithmetic_overflow"):
            memory.chunk_rounded_union([self.r(size=memory.U64_MAX), self.r(size=memory.U64_MAX, allocation=2)], chunk_bytes=memory.U64_MAX)

    def test_property_random_unaligned_ranges_against_byte_set(self):
        random_source = random.Random(424242)
        for _ in range(200):
            chunk = random_source.randrange(1, 33)
            base = random_source.randrange(chunk)
            ranges = []
            expected = set()
            for _ in range(random_source.randrange(1, 12)):
                offset = random_source.randrange(128)
                length = random_source.randrange(129 - offset)
                ranges.append(self.r(offset, length, size=128, base=base))
                expected.update((base + value) // chunk for value in range(offset, offset + length))
            result = memory.chunk_rounded_union(ranges, chunk_bytes=chunk)
            self.assertEqual(result["chunk_count_upper_bound"], len(expected))
            self.assertEqual(result, memory.chunk_rounded_union(list(reversed(ranges)), chunk_bytes=chunk))

    def test_raw_addresses_not_in_result(self):
        result = memory.chunk_rounded_union([self.r(base=3)], chunk_bytes=16)
        serialized = json.dumps(result)
        self.assertNotIn("0x", serialized)
        self.assertNotIn("native_handle", serialized)
        self.assertFalse(result["runtime_binding_proven"])
        self.assertFalse(result["admission_ready"])


class SourceVerificationTests(unittest.TestCase):
    def test_empty_source_root_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(memory.MemoryRuleError, "required_source_unverified"):
                memory.SourceMemoryRules(directory)

    def test_tampered_source_status_is_not_accepted(self):
        entries = [{"path": path} for path in memory.REQUIRED_SOURCES]
        with mock.patch.object(memory.sources, "load_profile", return_value=({"files": entries}, "a" * 64)), \
             mock.patch.object(memory.sources, "verify_sources", return_value=({}, [{"status": "hash_mismatch"}])):
            with self.assertRaisesRegex(memory.MemoryRuleError, "required_source_unverified"):
                memory.SourceMemoryRules("unused")

    def test_public_rules_always_preserve_source_only_scope(self):
        provenance = {"upstream_commit": memory.sources.COMMIT, "sources": [{"scope": "test_fixture"}]}
        with mock.patch.object(memory, "_verified_provenance", return_value=provenance):
            model = memory.SourceMemoryRules("unused")
        for result in (model.quantize_q8_1(**quantize_args()), model.stream_k_fixup(**fixup_args())):
            self.assertEqual(result["source_provenance"], provenance)
            self.assertEqual(result["evidence_scope"], "source_model_only")
            self.assertFalse(result["runtime_binding_proven"])
            self.assertFalse(result["device_ordering_proven"])
            self.assertFalse(result["admission_ready"])

    def test_returned_provenance_cannot_mutate_later_results(self):
        provenance = {"upstream_commit": memory.sources.COMMIT, "sources": [{"scope": "test_fixture"}]}
        with mock.patch.object(memory, "_verified_provenance", return_value=provenance):
            model = memory.SourceMemoryRules("unused")
        first = model.quantize_q8_1(**quantize_args())
        first["source_provenance"]["sources"][0]["scope"] = "changed"
        provenance["sources"][0]["scope"] = "changed_again"
        second = model.stream_k_fixup(**fixup_args())
        self.assertEqual(second["source_provenance"]["sources"][0]["scope"], "test_fixture")


if __name__ == "__main__":
    unittest.main()
