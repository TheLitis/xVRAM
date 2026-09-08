"""No-driver declared-scalar arithmetic tests, never captured-launch evidence."""

import random
import unittest
from unittest import mock

from xvram import compat_audit_kernel_ranges as ranges
from xvram.compat_audit_memory import MemoryRuleError


def set_args(**changes):
    result = dict(shape=(4, 2, 1, 1), index_shape=(2, 1, 1, 1),
                  source_strides=(6, 12, 12), index_strides=(1, 2, 2),
                  destination_strides=(8, 32, 32),
                  descriptors=tuple(ranges.fastdiv_descriptor(d) for d in (4, 2, 1, 1, 1)),
                  indices={0: 3, 8: 1}, grid=(1, 1, 1), block=(32, 1, 1))
    result.update(changes)
    return result


def get_args(**changes):
    result = dict(columns=4, index_shape=(2, 1, 1), source_byte_strides=(32, 128, 128),
                  index_strides=(1, 2, 2), destination_strides=(6, 12, 12),
                  descriptor=ranges.fastdiv_descriptor(1), indices={0: 3, 4: 1},
                  grid=(2, 1, 1), block=(32, 1, 1))
    result.update(changes)
    return result


def rope_args(**changes):
    result = dict(shape=(8, 2, 2, 1), source_strides=(8, 16, 32),
                  destination_strides=(8, 16, 32), n_dims=4, n_offs=2,
                  grid=(4, 1, 1), block=(1, 32, 1))
    result.update(changes)
    return result


def table_args(**changes):
    result = dict(ne12=2, ne13=2, ne23=4, source0_byte_strides=(100, 1000),
                  source1_byte_strides=(200, 2000), destination_byte_strides=(300, 3000),
                  broadcast_ratios=(2, 1), grid=(1, 1, 1), block=(16, 16, 1))
    result.update(changes)
    return result


def intervals(result, operand, mode):
    return [(r["offset_bytes"], r["length_bytes"]) for r in result["ranges"]
            if r["operand"] == operand and r["mode"] == mode]


class FastdivTests(unittest.TestCase):
    def test_matches_pinned_unsigned_arithmetic_inside_bounded_domain(self):
        rng = random.Random(817)
        for divisor in [1, 2, 3, 7, 32, 127, 2048, 2**30, 2**31 - 1]:
            mul, shift, actual = ranges.fastdiv_descriptor(divisor)
            self.assertEqual(actual, divisor)
            for value in [0, 1, 2**31 - 1] + [rng.randrange(2**31) for _ in range(80)]:
                hi = (value * mul) >> 32
                self.assertLessEqual(hi + value, 2**32 - 1)
                self.assertEqual((hi + value) >> shift, value // divisor)

    def test_rejects_outside_domain_and_booleans(self):
        for value in [0, -1, True, 2**31, 1.0]:
            with self.subTest(value=value), self.assertRaises(MemoryRuleError):
                ranges.fastdiv_descriptor(value)


class SetRowsTests(unittest.TestCase):
    def test_sparse_destination_and_source_padding(self):
        result = ranges.set_rows(**set_args())
        self.assertEqual(intervals(result, "src0", "read"), [(0, 16), (24, 16)])
        self.assertEqual(intervals(result, "src1", "read"), [(0, 16)])
        self.assertEqual(intervals(result, "dst", "write"), [(16, 8), (48, 8)])
        self.assertFalse(result["memory_bounds_proven"])
        self.assertFalse(result["admission_ready"])

    def test_index_broadcast_and_channels(self):
        result = ranges.set_rows(**set_args(shape=(4, 2, 2, 1),
            descriptors=tuple(ranges.fastdiv_descriptor(d) for d in (4, 2, 2, 1, 1))))
        self.assertEqual(intervals(result, "src1", "read"), [(0, 16)])
        self.assertEqual(intervals(result, "dst", "write"), [(16, 8), (48, 8), (80, 8), (112, 8)])

    def test_repeated_writes_are_rejected_not_hidden_by_union(self):
        with self.assertRaisesRegex(MemoryRuleError, "overlapping_destination"):
            ranges.set_rows(**set_args(indices={0: 1, 8: 1}))

    def test_snapshot_missing_negative_unaligned_or_boolean(self):
        for snapshot in [{0: 1}, {0: -1, 8: 2}, {1: 1, 8: 2}, {0: True, 8: 2}]:
            with self.subTest(snapshot=snapshot), self.assertRaises(MemoryRuleError):
                ranges.set_rows(**set_args(indices=snapshot))

    def test_wrong_fastdiv_is_not_accepted_from_divisor_only(self):
        descriptors = list(set_args()["descriptors"])
        descriptors[0] = (0, 0, 4)
        with self.assertRaisesRegex(MemoryRuleError, "fastdiv_descriptor_mismatch"):
            ranges.set_rows(**set_args(descriptors=descriptors))

    def test_geometry_and_signed_cast_bounds(self):
        for changes in [dict(grid=(2, 1, 1)), dict(block=(32, 2, 1)),
                        dict(shape=(2**30, 2, 1, 1)), dict(source_strides=(-1, 12, 12)),
                        dict(indices={0: 2**63 - 1, 8: 1}), dict(index_shape=(1, 1, 1, 1))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.set_rows(**set_args(**changes))

    def test_random_rows_match_scalar_source_oracle(self):
        rng = random.Random(21)
        for _ in range(80):
            n0, n1 = rng.randrange(1, 20), rng.randrange(1, 12)
            chosen = rng.sample(range(32), n1)
            args = set_args(shape=(n0, n1, 1, 1), index_shape=(n1, 1, 1, 1),
                            source_strides=(n0 + 2, 1000, 1000),
                            destination_strides=(n0 + 3, 1000, 1000),
                            descriptors=tuple(ranges.fastdiv_descriptor(d) for d in (n0, n1, 1, 1, 1)),
                            indices={i * 8: row for i, row in enumerate(chosen)},
                            grid=((n0*n1 + 31)//32, 1, 1))
            actual = ranges.set_rows(**args)
            expected = {2*(row*(n0+3)+col)+byte for row in chosen for col in range(n0) for byte in range(2)}
            observed = {i for first, length in intervals(actual, "dst", "write") for i in range(first, first+length)}
            self.assertEqual(observed, expected)


class GetRowsTests(unittest.TestCase):
    def test_source_strides_are_bytes_destination_elements(self):
        result = ranges.get_rows_float(**get_args())
        self.assertEqual(intervals(result, "src0", "read"), [(32, 16), (96, 16)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 16), (24, 16)])

    def test_duplicate_reads_allowed(self):
        result = ranges.get_rows_float(**get_args(indices={0: 3, 4: 3}))
        self.assertEqual(intervals(result, "src0", "read"), [(96, 16)])

    def test_z_fastdiv_and_index_striding(self):
        result = ranges.get_rows_float(**get_args(index_shape=(1, 2, 2),
            indices={0: 0, 8: 1, 16: 2}, index_strides=(1, 2, 2),
            descriptor=ranges.fastdiv_descriptor(2), grid=(1, 1, 1),
            destination_strides=(4, 8, 16)))
        self.assertEqual(intervals(result, "src1", "read"), [(0, 4), (8, 4), (16, 4)])

    def test_rejects_missing_invalid_geometry_and_overflow(self):
        for changes in [dict(indices={}), dict(grid=(3, 1, 1)), dict(block=(32, 2, 1)),
                        dict(source_byte_strides=(31, 128, 128)), dict(columns=2**63 - 1),
                        dict(destination_strides=(0, 0, 0))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.get_rows_float(**get_args(**changes))


class RopeTests(unittest.TestCase):
    def test_all_columns_copy_outside_rotation(self):
        result = ranges.rope_neox(**rope_args())
        self.assertEqual(intervals(result, "x", "read"), [(0, 128)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 128)])
        self.assertEqual(intervals(result, "pos", "read"), [(0, 8)])

    def test_inplace_skips_unrotated_columns(self):
        result = ranges.rope_neox(**rope_args(inplace=True))
        self.assertEqual(intervals(result, "x", "read"), [(8, 16), (40, 16), (72, 16), (104, 16)])

    def test_fused_kv_indices_and_fp16_destination(self):
        result = ranges.rope_neox(**rope_args(set_rows_stride=16, indices={0: 3, 8: 1}, destination_bytes=2))
        self.assertEqual(intervals(result, "dst", "write"), [(32, 32), (96, 32)])
        self.assertEqual(intervals(result, "row_indices", "read"), [(0, 16)])

    def test_unguarded_rows_reject_rounded_up_grid(self):
        with self.assertRaisesRegex(MemoryRuleError, "invalid_rope_geometry"):
            ranges.rope_neox(**rope_args(grid=(5, 1, 1)))

    def test_fused_duplicate_writes_and_int_narrowing_rejected(self):
        for changes in [dict(set_rows_stride=16, indices={0: 1, 8: 1}),
                        dict(set_rows_stride=16, indices={0: 2**31, 8: 1}),
                        dict(shape=(8, 2, 2, 2), grid=(8, 1, 1),
                             set_rows_stride=16, indices={0: 1, 8: 2})]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.rope_neox(**rope_args(**changes))

    def test_unsupported_flags_and_shapes_fail_closed(self):
        for changes in [dict(n_dims=3), dict(n_offs=3), dict(n_dims=8),
                        dict(has_freq_factors=True), dict(inplace=1), dict(indices={0: 1}),
                        dict(source_bytes=2), dict(source_strides=(2**31, 1, 1))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.rope_neox(**rope_args(**changes))

    def test_scalar_pair_formula_with_offsets(self):
        for n0 in (8, 12, 24):
            for n_dims in (2, 4, 6):
                for n_offs in range(0, n0 - n_dims + 1, 2):
                    for inplace in (False, True):
                        result = ranges.rope_neox(**rope_args(shape=(n0, 1, 1, 1),
                            n_dims=n_dims, n_offs=n_offs, inplace=inplace,
                            source_strides=(n0, n0, n0), destination_strides=(n0, n0, n0), grid=(1, 1, 1)))
                        expected = set()
                        for i0 in range(0, n0, 2):
                            if i0 < n_offs or i0 >= n_offs+n_dims:
                                if not inplace:
                                    expected.update((i0, i0+1))
                            else:
                                expected.update((i0//2+n_offs//2, i0//2+n_offs//2+n_dims//2))
                        actual = {i//4 for first, length in intervals(result, "x", "read") for i in range(first, first+length)}
                        self.assertEqual(actual, expected)


class PointerTableTests(unittest.TestCase):
    def test_only_pointer_tables_are_memory_accesses(self):
        result = ranges.batched_pointer_tables(**table_args())
        self.assertEqual(intervals(result, "ptrs_src", "write"), [(0, 64)])
        self.assertEqual(intervals(result, "ptrs_dst", "write"), [(0, 32)])
        self.assertFalse(any(r["mode"] == "read" for r in result["ranges"]))
        self.assertEqual(result["pointer_targets"][3],
                         {"batch": 3, "src0_offset_bytes": 1000, "src1_offset_bytes": 2200, "dst_offset_bytes": 3300})
        self.assertFalse(result["consumer_memory_bounds_proven"])

    def test_table_count_geometry_zero_ratio_and_overflow(self):
        for changes in [dict(ne23=5), dict(grid=(2, 1, 1)), dict(broadcast_ratios=(0, 1)),
                        dict(ne12=100_001), dict(source1_byte_strides=(2**64-1, 1))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.batched_pointer_tables(**table_args(**changes))


class DirectKernelTests(unittest.TestCase):
    def rms_args(self, **changes):
        result = dict(columns=1024, shape=(2, 2, 1), source_strides=(1040, 2080, 4160),
                      grid=(2, 2, 1), block=(1024, 1, 1), shared_bytes=128)
        result.update(changes)
        return result

    def mmq_args(self, **changes):
        result = dict(valid_columns=132, padded_columns=256, shape=(2, 2, 1),
                      source_strides=(136, 272, 544), ds_layout=0,
                      grid=(2, 2, 2), block=(32, 1, 1))
        result.update(changes)
        return result

    def matvec_args(self, **changes):
        result = dict(columns2=64, shape=(3, 2, 1), source_strides=(128, 384, 768),
                      vector_strides=(64, 128, 256), destination_strides=(3, 3, 6),
                      channel_ratio=1, sample_ratio=1,
                      descriptors=(ranges.fastdiv_descriptor(1),)*2, grid=(3, 2, 1),
                      block=(64, 1, 1), compiled_block_size=64, shared_bytes=128)
        result.update(changes)
        return result

    def test_rms_strided_source_contiguous_output(self):
        result = ranges.rms_norm(**self.rms_args())
        self.assertEqual(intervals(result, "x", "read"), [(i*4160, 4096) for i in range(4)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 16384)])

    def test_rms_broadcast_multiplier(self):
        result = ranges.rms_norm(**self.rms_args(multiply=True, multiplier_shape=(32, 1, 1, 1),
            multiplier_strides=(0, 0, 0), multiplier_descriptors=tuple(ranges.fastdiv_descriptor(i) for i in (32, 1, 1, 1))))
        self.assertEqual(intervals(result, "mul", "read"), [(0, 128)])

    def test_rms_invalid_shared_geometry_and_scalar_bounds(self):
        for changes in [dict(shared_bytes=64), dict(block=(256, 1, 1)), dict(columns=2**31-1),
                        dict(multiply=True), dict(multiplier_shape=(1, 1, 1, 1))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.rms_norm(**self.rms_args(**changes))

    def test_mmq_quantize_transposed_contiguous_layouts(self):
        for layout in (0, 1):
            result = ranges.quantize_mmq_q8_1(**self.mmq_args(ds_layout=layout))
            self.assertEqual(intervals(result, "x", "read"), [(i*544, 528) for i in range(4)])
            self.assertEqual(intervals(result, "vy", "write"), [(0, 1152)])
            self.assertEqual(result["channel_pitch_blocks"], 4)

    def test_mmq_grid_padding_changes_channel_stride_not_reads(self):
        result = ranges.quantize_mmq_q8_1(**self.mmq_args(valid_columns=4, padded_columns=128,
            grid=(2, 1, 2), block=(64, 1, 1)))
        self.assertEqual(intervals(result, "vy", "write"), [(0, 288), (576, 288)])

    def test_mmq_indexed_rows_and_alignment(self):
        result = ranges.quantize_mmq_q8_1(**self.mmq_args(indices={0: 2, 4: 0}))
        self.assertEqual(intervals(result, "ids", "read"), [(0, 8)])
        for changes in [dict(valid_columns=133), dict(source_strides=(135, 272, 544)),
                        dict(ds_layout=2), dict(padded_columns=255), dict(indices={})]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.quantize_mmq_q8_1(**self.mmq_args(**changes))

    def test_fp16_matvec_sizes_and_broadcast_reads(self):
        result = ranges.matvec_f16(**self.matvec_args())
        self.assertEqual(intervals(result, "x", "read"), [(0, 1536)])
        self.assertEqual(intervals(result, "y", "read"), [(0, 1024)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 24)])

    def test_fp16_matvec_rejects_indirect_and_int_product_overflow(self):
        for changes in [dict(ids_null=False), dict(shared_bytes=0), dict(columns2=2**31-1),
                        dict(source_strides=(2**31-1, 384, 768)), dict(vector_strides=(64, 127, 256))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.matvec_f16(**self.matvec_args(**changes))


class QuantizedMatvecTests(unittest.TestCase):
    def args(self, **changes):
        result = dict(quant_type=12, columns=256, output_rows=3, output_columns=2,
                      channels=1, samples=1, source_strides=(1, 4, 4),
                      vector_strides=(8, 16, 16), destination_strides=(3, 6, 6),
                      channel_ratio=1, sample_ratio=1,
                      descriptors=(ranges.fastdiv_descriptor(1),)*2,
                      grid=(2, 1, 1), block=(32, 4, 1), compiled_arch=860)
        result.update(changes)
        return result

    def test_padded_source_row_is_read_even_when_destination_guard_skips_it(self):
        result = ranges.matvec_quantized(**self.args())
        self.assertEqual(intervals(result, "x", "read"), [(0, 4*144)])
        self.assertEqual(intervals(result, "y", "read"), [(0, 16*36)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 6*4)])
        self.assertEqual(result["padded_source_rows"], 1)

    def test_q6_block_size_and_fused_bias_gate(self):
        result = ranges.matvec_quantized(**self.args(quant_type=14, output_columns=1,
            grid=(3, 1, 1), has_fusion=True, gate=True, bias=True, gate_bias=True))
        self.assertEqual(intervals(result, "x", "read"), [(0, 3*210)])
        self.assertEqual(intervals(result, "gate", "read"), [(0, 3*210)])
        self.assertEqual(intervals(result, "x_bias", "read"), [(0, 12)])
        self.assertEqual(intervals(result, "gate_bias", "read"), [(0, 12)])

    def test_rejects_wrong_compiled_table_ids_or_strides(self):
        for changes in [dict(compiled_arch=750), dict(ids_null=False), dict(columns=257),
                        dict(has_fusion=True), dict(gate=True), dict(block=(32, 2, 1)),
                        dict(destination_strides=(4, 8, 8)), dict(source_strides=(2**32-1, 0, 0))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.matvec_quantized(**self.args(**changes))


class CaptureLayoutTests(unittest.TestCase):
    def test_ordinals_types_and_lengths(self):
        lengths = {"set_rows": 22, "get_rows_float": 15, "rope_neox": 23,
                   "quantize_q8_1": 9, "quantize_mmq_q8_1": 11, "rms_norm_f32": 23,
                   "batched_pointer_tables": 16, "mul_mat_vec_f": 19, "mul_mat_vec_q": 19,
                   "mul_mat_q": 24, "mul_mat_q_stream_k_fixup": 13, "cpy_scalar": 17,
                   "convert_unary": 9, "unary_gated_op_kernel": 7, "k_bin_bcast": 23,
                   "soft_max_f32": 5, "mul_mat_f": 20}
        for family, count in lengths.items():
            result = ranges.capture_layout(family)
            self.assertEqual(len(result["arguments"]), count)
            self.assertEqual([arg["ordinal"] for arg in result["arguments"]], list(range(count)))
            self.assertFalse(result["binary_abi_proven"])
        self.assertEqual(ranges.capture_layout("set_rows")["arguments"][3]["name"], "ne_total")
        self.assertEqual(ranges.capture_layout("mul_mat_vec_q")["arguments"][3]["type"], "fusion_struct")

    def test_unknown_family_does_not_guess(self):
        with self.assertRaises(MemoryRuleError):
            ranges.capture_layout("opaque_kernel")


class MMQStreamKTests(unittest.TestCase):
    def args(self, **changes):
        result = dict(stage="main", quant_type=12, tile_columns=40, columns=768,
                      output_rows=256, output_columns=1, vector_columns=1, channels=1, samples=1,
                      source_strides=(3, 768, 768), vector_strides=(216, 216),
                      destination_strides=(256, 256, 256), channel_ratio=1, sample_ratio=1,
                      descriptors=tuple(ranges.fastdiv_descriptor(i) for i in (3, 1, 1, 1, 1, 1)),
                      grid=(4, 1, 1), block=(32, 8, 1), shared_bytes=45216, compiled_arch=860)
        result.update(changes)
        return result

    def test_j40_y_load_rounds_up_all_lanes_not_logical_columns(self):
        result = ranges.mmq_stream_k(**self.args())
        self.assertEqual(result["y_tile_load_bytes"], 6144)
        # Last half-block begins at 5*36 int and still reads 1536 int.
        self.assertEqual(intervals(result, "y", "read"), [(0, (5*36+1536)*4)])
        self.assertEqual(intervals(result, "x", "read"), [(0, 256*3*144)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 1024)])

    def test_main_and_fixup_slots_reconcile(self):
        main = ranges.mmq_stream_k(**self.args())
        fixup = ranges.mmq_stream_k(**self.args(stage="fixup", grid=(4, 4, 1), block=(32, 4, 1), shared_bytes=0))
        self.assertEqual(main["fixup_written_slots"], [0, 2])
        self.assertEqual(main["global_scratch_bytes"], 4*40*128*4)
        self.assertEqual(intervals(main, "tmp_fixup", "write"), intervals(fixup, "tmp_fixup", "read"))
        self.assertEqual(intervals(fixup, "dst", "read"), [(0, 1024)])
        self.assertEqual(intervals(fixup, "dst", "write"), [(0, 1024)])
        self.assertTrue(fixup["requires_matching_main_content_generation"])

    def test_empty_blocks_and_repeated_partial_contributors(self):
        for blocks in (1, 2, 3, 4, 5, 6, 7, 16, 20):
            main = ranges.mmq_stream_k(**self.args(grid=(blocks, 1, 1)))
            fixup = ranges.mmq_stream_k(**self.args(stage="fixup", grid=(blocks, 4, 1), block=(32, 4, 1), shared_bytes=0))
            self.assertEqual(intervals(main, "tmp_fixup", "write"), intervals(fixup, "tmp_fixup", "read"))
            self.assertEqual(intervals(main, "dst", "write"), [(0, 1024)])

    def test_j128_q6_full_tile(self):
        result = ranges.mmq_stream_k(**self.args(quant_type=14, tile_columns=128,
            output_columns=128, vector_columns=128, shared_bytes=57856))
        self.assertEqual(result["y_tile_load_bytes"], 128*144)
        self.assertEqual(intervals(result, "x", "read"), [(0, 256*3*210)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 256*128*4)])

    def test_no_fallback_geometry_or_unknown_ids_never_admitted(self):
        for changes in [dict(output_rows=255), dict(ids_null=False), dict(compiled_arch=750),
                        dict(shared_bytes=0), dict(block=(32, 4, 1)), dict(columns=767),
                        dict(output_columns=2), dict(stage="other"),
                        dict(source_strides=(2**31-1, 1, 1))]:
            with self.subTest(changes=changes), self.assertRaises(MemoryRuleError):
                ranges.mmq_stream_k(**self.args(**changes))


class OtherDirectTests(unittest.TestCase):
    def test_reshape_copy_changes_row_boundaries_preserves_padding(self):
        result = ranges.copy_f32(elements=12, source_shape=(4, 3, 1, 1), destination_shape=(6, 2, 1, 1),
            source_byte_strides=(4, 20, 60, 60), destination_byte_strides=(4, 32, 64, 64),
            grid=(1, 1, 1), block=(32, 1, 1))
        self.assertEqual(intervals(result, "cx", "read"), [(0, 16), (20, 16), (40, 16)])
        self.assertEqual(intervals(result, "cdst", "write"), [(0, 24), (32, 24)])

    def test_copy_rejects_unmodelled_column_stride_and_shape_mismatch(self):
        args = dict(elements=12, source_shape=(4, 3, 1, 1), destination_shape=(6, 2, 1, 1),
            source_byte_strides=(4, 20, 60, 60), destination_byte_strides=(4, 32, 64, 64),
            grid=(1, 1, 1), block=(32, 1, 1))
        for update in (dict(elements=11), dict(source_byte_strides=(8, 40, 120, 120)), dict(grid=(2, 1, 1))):
            with self.assertRaises(MemoryRuleError):
                ranges.copy_f32(**{**args, **update})

    def test_conversion_tails_and_type_sizes(self):
        for source_bytes, destination_bytes in ((2, 4), (4, 2)):
            result = ranges.convert_f16_f32(shape=(35, 2, 2, 1), source_strides=(40, 80, 160),
                source_bytes=source_bytes, destination_bytes=destination_bytes, descriptor=ranges.fastdiv_descriptor(2),
                grid=(2, 2, 2), block=(32, 1, 1))
            self.assertEqual(intervals(result, "vx", "read"), [(i*40*source_bytes, 35*source_bytes) for i in range(4)])
            self.assertEqual(intervals(result, "y", "write"), [(0, 140*destination_bytes)])

    def test_gated_final_partial_row_and_separate_gate_stride(self):
        result = ranges.unary_gated(elements=10, row_columns=4, source_row_stride=8, gate_row_stride=4,
                                    grid=(1, 1, 1), block=(32, 1, 1))
        self.assertEqual(intervals(result, "x", "read"), [(0, 16), (32, 16), (64, 8)])
        self.assertEqual(intervals(result, "g", "read"), [(0, 40)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 40)])

    def binary_args(self, **changes):
        result = dict(shape=(8, 2, 2, 1), broadcast_shape=(2, 1, 1, 1),
            source_strides=(1, 8, 16, 32), broadcast_strides=(1, 0, 0, 0),
            destination_strides=(8, 16, 32),
            descriptors=tuple(ranges.fastdiv_descriptor(v) for v in (1, 2, 1, 1, 1)),
            grid=(1, 2, 2), block=(32, 1, 1))
        result.update(changes)
        return result

    def test_binary_pack_pointer_not_unused_formal_argument(self):
        result = ranges.binary_broadcast(**self.binary_args())
        self.assertEqual(intervals(result, "src1s0", "read"), [(0, 8)])
        self.assertEqual(intervals(result, "src0", "read"), [(0, 128)])
        self.assertFalse(any(row["operand"] == "src1" for row in result["ranges"]))

    def test_binary_missing_source0_and_read_holes(self):
        result = ranges.binary_broadcast(**self.binary_args(source0_present=False, broadcast_strides=(3, 0, 0, 0)))
        self.assertEqual(intervals(result, "src1s0", "read"), [(0, 16)])
        self.assertFalse(intervals(result, "src0", "read"))

    def test_binary_wrong_geometry_and_alias_writes_rejected(self):
        for changes in (dict(grid=(1, 1, 2)), dict(destination_strides=(0, 0, 0)), dict(source0_present=1)):
            with self.assertRaises(MemoryRuleError):
                ranges.binary_broadcast(**self.binary_args(**changes))

    def softmax_args(self, **changes):
        result = dict(shape=(256, 2, 2, 1), mask_shape=(1, 1), mask_byte_strides=(1024, 0, 0),
                      grid=(2, 2, 1), block=(256, 1, 1), shared_bytes=1152)
        result.update(changes)
        return result

    def test_softmax_template_columns_mask_broadcast_and_sinks(self):
        result = ranges.softmax_256(**self.softmax_args(sinks_present=True))
        self.assertEqual(intervals(result, "x", "read"), [(0, 4096)])
        self.assertEqual(intervals(result, "mask", "read"), [(0, 2048)])
        self.assertEqual(intervals(result, "sinks", "read"), [(0, 8)])

    def test_softmax_null_mask_still_validates_modulo_divisors(self):
        result = ranges.softmax_256(**self.softmax_args(mask_present=False))
        self.assertFalse(intervals(result, "mask", "read"))
        with self.assertRaises(MemoryRuleError):
            ranges.softmax_256(**self.softmax_args(mask_present=False, mask_shape=(0, 1)))

    def test_softmax_does_not_trust_scalar_ncols_over_template(self):
        for update in (dict(shape=(128, 2, 2, 1)), dict(block=(128, 1, 1)), dict(shared_bytes=1024)):
            with self.assertRaises(MemoryRuleError):
                ranges.softmax_256(**self.softmax_args(**update))

    def matf_args(self, **changes):
        result = dict(columns2=64, output_rows=64, channels=1, samples=1,
            source_strides=(64, 4096, 4096), vector_strides=(64, 256, 256),
            destination_strides=(64, 128, 128), channel_ratio=1, sample_ratio=1,
            grid=(2, 1, 1), block=(32, 1, 1), shared_bytes=2304, compiled_arch=860, nwarps=1)
        result.update(changes)
        return result

    def test_matf_half2_source_float2_vector_two_output_columns(self):
        result = ranges.matf_half2(**self.matf_args())
        self.assertEqual(intervals(result, "x", "read"), [(0, 64*64*4)])
        self.assertEqual(intervals(result, "y", "read"), [(0, 2*64*8)])
        self.assertEqual(intervals(result, "dst", "write"), [(0, 2*64*4)])

    def test_matf_two_warps_same_global_ranges(self):
        one = ranges.matf_half2(**self.matf_args())
        two = ranges.matf_half2(**self.matf_args(nwarps=2, block=(32, 2, 1), shared_bytes=4608))
        self.assertEqual(one["ranges"], two["ranges"])

    def test_matf_rejects_unmodelled_tail_warp_or_partial_output_rows(self):
        for update in (dict(columns2=63), dict(output_rows=63), dict(compiled_arch=750),
                       dict(shared_bytes=0), dict(ids_null=False), dict(destination_strides=(32, 64, 64))):
            with self.assertRaises(MemoryRuleError):
                ranges.matf_half2(**self.matf_args(**update))


class ProvenanceTests(unittest.TestCase):
    def test_required_source_hash_failure_rejects(self):
        profile, _ = ranges.sources.load_profile()
        with mock.patch.object(ranges.sources, "verify_sources", return_value=({}, [{"status": "hash_mismatch"}])):
            with self.assertRaisesRegex(MemoryRuleError, "required_source_unverified"):
                ranges.SourceKernelRanges("unused")
        profile["files"] = []
        with mock.patch.object(ranges.sources, "load_profile", return_value=(profile, "a"*64)):
            with self.assertRaisesRegex(MemoryRuleError, "required_source_not_pinned"):
                ranges.SourceKernelRanges("unused")

    def test_verified_model_still_cannot_certify_native_launch(self):
        profile, _ = ranges.sources.load_profile()
        results = [{**entry, "status": "verified"} for entry in profile["files"] if entry["path"] in ranges.SOURCE_LINES]
        with mock.patch.object(ranges.sources, "verify_sources", return_value=({}, results)):
            model = ranges.SourceKernelRanges("unused")
        result = model.evaluate("set_rows", **set_args())
        self.assertEqual(result["source_provenance"]["upstream_commit"], ranges.sources.COMMIT)
        self.assertFalse(result["runtime_binding_proven"])
        self.assertFalse(result["memory_bounds_proven"])
        with self.assertRaisesRegex(MemoryRuleError, "unsupported_kernel_rule"):
            model.evaluate("opaque_cublas_private_kernel")


if __name__ == "__main__":
    unittest.main()
