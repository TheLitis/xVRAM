"""Synthetic typed-argument/source arithmetic; not hardware acceptance."""
import unittest
from unittest import mock

from xvram import compat_audit_captured_ranges as captured
from xvram import compat_audit_kernel_arguments as arguments
from xvram import compat_audit_kernel_ranges as ranges
from test_compat_audit_kernel_arguments import fixture


def call(kid, scalars, grid, block=(32, 1, 1), shared=0, nulls=()):
    rows = fixture(kid)
    rows[1].update({"grid_"+axis: value for axis, value in zip("xyz", grid)})
    rows[1].update({"block_"+axis: value for axis, value in zip("xyz", block)})
    rows[1]["shared_bytes"] = shared
    for row in rows:
        if row["kind"] == "field" and row["value_type"] == "ptr" and row["argument"] in nulls:
            for key in set(row) - (arguments.COMMON | {"ordinal", "argument", "field", "value_type", "memory_bounds_proven"}):
                del row[key]
            row["resolution"] = "null"
        if row["kind"] != "field" or row["argument"] not in scalars:
            continue
        value = scalars[row["argument"]]
        if isinstance(value, dict):
            if row["field"] not in value:
                continue
            value = value[row["field"]]
        row["value"] = value["xyz".index(row["field"])] if type(value) is tuple else value
    _, calls = arguments.analyze_rows(rows)
    return calls[1]


def model():
    # Exercise the real verified-model constructor with explicit test fixtures
    # for file hashing. Hardware runs use real, unpatched source verification.
    profile, _ = ranges.sources.load_profile()
    results = [{**entry, "status": "verified"} for entry in profile["files"] if entry["path"] in ranges.SOURCE_LINES]
    with mock.patch.object(ranges.sources, "verify_sources", return_value=({}, results)), \
            mock.patch.object(ranges, "verify_extra_sources", return_value={"test_fixture": True}):
        return ranges.SourceKernelRanges("fixture", "fixture-extra")


class CapturedRangeTests(unittest.TestCase):
    def test_rope_position_values_are_not_needed_for_addresses(self):
        values = dict(ne00=128, ne01=40, ne02=2, s01=128, s02=5120, s03=10240,
                      s1=128, s2=5120, s3=10240, n_dims=128, n_offs=0, inplace=False, set_rows_stride=0)
        for kid in (34, 35):
            c = call(kid, values, (80, 1, 1), (1, 256, 1))
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"], result)
            self.assertFalse(result["position_values_affect_addresses"])
            self.assertFalse(result["position_contents_needed_for_address_ranges"])
            self.assertEqual(next(r["length_bytes"] for r in result["ranges"] if r["operand"] == "pos"), 8)
            self.assertEqual({r["operand"] for r in result["ranges"]}, {"x", "pos", "dst"})
            # has_ff=false excludes that pointer regardless of its bytes;
            # ordinary set_rows_stride=0 excludes row_indices altogether.
            self.assertFalse(result["memory_bounds_proven"])
        c = call(34, {**values, "set_rows_stride": 1024}, (80, 1, 1), (1, 256, 1))
        self.assertEqual(captured.evaluate_call(c, model())["reason"], "captured_rope_row_indices_generation_required")

    def test_rope_shape_comes_from_exact_launch_rows_and_inplace_interval(self):
        values = dict(ne00=128, ne01=2, ne02=2, s01=128, s02=256, s03=512,
                      s1=128, s2=256, s3=512, n_dims=64, n_offs=32, inplace=True, set_rows_stride=0)
        c = call(35, values, (4, 1, 1), (1, 64, 1))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"])
        x = [r for r in result["ranges"] if r["operand"] == "x"]
        self.assertEqual([(r["offset_bytes"], r["length_bytes"]) for r in x],
                         [(128, 256), (640, 256), (1152, 256), (1664, 256)])
        bad = call(35, values, (5, 1, 1), (1, 64, 1))
        self.assertEqual(captured.evaluate_call(bad, model())["reason"], "captured_dimension_not_exact")

    def test_pointer_producer_symbolic_affine_slots_not_consumer_bounds(self):
        values = dict(ne12=4, ne13=2, ne23=8, nb02=512, nb03=8192, nb12=1024, nb13=4096,
                      nbd2=2048, nbd3=8192, r2=2, r3=1)
        c = call(23, values, (1, 1, 1), (16, 16, 1))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"])
        self.assertEqual({r["operand"] for r in result["ranges"]}, {"ptrs_src", "ptrs_dst"})
        symbolic = result["symbolic_pointer_writes"]
        self.assertEqual(len(symbolic), 24)
        src0 = [r for r in symbolic if r["target_operand"] == "src0"]
        self.assertEqual([r["target_offset_bytes"] for r in src0], [0, 0, 512, 512, 8192, 8192, 8704, 8704])
        self.assertEqual([r["table_offset_bytes"] for r in src0], list(range(0, 64, 8)))
        src1 = [r for r in symbolic if r["target_operand"] == "src1"]
        self.assertEqual([r["table_offset_bytes"] for r in src1], list(range(64, 128, 8)))
        self.assertTrue(all(r["pointee_bytes_accessed_by_producer"] == 0 for r in symbolic))
        self.assertFalse(result["table_content_generation_proven"])
        self.assertFalse(result["consumer_memory_bounds_proven"])
        self.assertTrue(any(r["target_containment"] == "outside_observed_allocation" for r in symbolic))
        self.assertFalse(result["memory_bounds_proven"])

    def test_fp16_and_quantized_matvec_scalar_units_and_padding(self):
        base = dict(columns=256, stride_row_x=1, stride_col_y=8, stride_col_dst=3,
                    stride_channel_x=4, stride_channel_y=16, stride_channel_dst=6,
                    stride_sample_x=4, stride_sample_y=16, stride_sample_dst=6, ids_stride=0,
                    # The actual no-ID pilot passes zero for this dead formal
                    # descriptor; it must not become a fabricated dimension.
                    nchannels_y_fd=(0, 0, 0), channel_ratio_fd=ranges.fastdiv_descriptor(1),
                    sample_ratio_fd=ranges.fastdiv_descriptor(1))
        for kid in (12, 13, 14, 15, 16, 17):
            with self.subTest(kid=kid):
                columns = 2 if kid in (14, 17) else 1
                c = call(kid, base, (2 if columns == 2 else 3, 1, 1), (32, 4, 1), nulls=("ids",))
                result = captured.evaluate_call(c, model())
                self.assertTrue(result["source_relative_ranges_derived"], result)
                x = next(r for r in result["ranges"] if r["operand"] == "x")
                self.assertEqual(x["length_bytes"], (4 if columns == 2 else 3)*(144 if kid < 15 else 210))
                self.assertFalse(result["memory_bounds_proven"])
                if kid in (13, 16):
                    self.assertIn("gate", {r["operand"] for r in result["ranges"]})
        for kid, threads in ((10, 128), (11, 64)):
            c = call(kid, {**base, "columns": 32, "stride_row_x": 64}, (3, 1, 1), (threads, 1, 1), 128, ("ids",))
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"], result)
            self.assertEqual(next(r["length_bytes"] for r in result["ranges"] if r["operand"] == "x"), 384)

    def test_mmq_main_uses_captured_k_descriptor_and_explicit_tail_load(self):
        values = dict(nrows_x=128, ncols_dst=8, ncols_y=8, stride_row_x=2, stride_col_dst=128,
                      stride_channel_x=256, stride_channel_y=1152, stride_channel_dst=1024,
                      stride_sample_x=256, stride_sample_y=1152, stride_sample_dst=1024)
        values.update({name: ranges.fastdiv_descriptor(size) for name, size in zip(
            ("blocks_per_ne00_fd", "channel_ratio_fd", "nchannels_y_fd", "sample_ratio_fd", "nsamples_y_fd", "ntx_fd"),
            (2, 1, 1, 1, 1, 1))})
        for kid in (30, 31, 32, 33):
            tile = 128 if kid in (30, 32) else 40
            shared = (tile + ((tile*36+255)//256)*256 + 128*76)*4
            c = call(kid, values, (3, 1, 1), (32, 8, 1), shared, ("ids_dst", "expert_bounds", "y_scale"))
            rule, scalar = captured.direct_scalars(c)
            self.assertEqual(scalar["columns"], 512)
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"], result)
            self.assertIn("tmp_fixup", {r["operand"] for r in result["ranges"]})
            self.assertFalse(result["source_semantics_bound_to_cubin"])

    def test_vocabulary_projection_uses_exact_analytic_contiguous_intervals(self):
        values = dict(columns=5120, stride_row_x=20, stride_col_y=160, stride_col_dst=152064,
                      stride_channel_x=3041280, stride_channel_y=160, stride_channel_dst=152064,
                      stride_sample_x=3041280, stride_sample_y=160, stride_sample_dst=152064,
                      ids_stride=0, nchannels_y_fd=(0, 0, 0), channel_ratio_fd=ranges.fastdiv_descriptor(1),
                      sample_ratio_fd=ranges.fastdiv_descriptor(1))
        c = call(15, values, (152064, 1, 1), (32, 4, 1), nulls=("ids",))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"])
        self.assertEqual(result["interval_strategy"], "analytical_contiguous_single_output")
        self.assertEqual(next(r["length_bytes"] for r in result["ranges"] if r["operand"] == "x"), 638668800)
        self.assertFalse(result["memory_bounds_proven"])
        c = call(15, {**values, "stride_row_x": 21}, (152064, 1, 1), (32, 4, 1), nulls=("ids",))
        self.assertEqual(captured.evaluate_call(c, model())["reason"], "captured_source_evaluator_row_capacity")

    def test_quantizers_use_exact_channel_grid_quotient(self):
        values = dict(ne00=128, ne0=128, ne1=2, ne2=1, s01=128, s02=256, s03=256,
                      n_expert_used=0, ne2_fd=ranges.fastdiv_descriptor(1))
        for kid in (18, 20, 21):
            grid = (1, 2, 1) if kid == 18 else (2, 1, 1)
            c = call(kid, values, grid, (128 if kid == 18 else 32, 1, 1), nulls=("ids",))
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"], result)
            self.assertEqual(next(r["length_bytes"] for r in result["ranges"] if r["operand"] == "vy"), 288)
        c = call(20, {**values, "ne2": 2}, (2, 1, 3), (32, 1, 1), nulls=("ids",))
        self.assertEqual(captured.evaluate_call(c, model())["reason"], "captured_dimension_not_exact")

    def test_softmax_struct_and_half2_compiled_outputs(self):
        params = dict(ne00=256, ne01=2, ne02=1, ne03=1, ne12=1, ne13=1, nb11=1024, nb12=2048, nb13=2048)
        c = call(7, dict(params=params), (2, 1, 1), (256, 1, 1), 1152, ("mask", "sinks"))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"], result)
        self.assertEqual({r["operand"] for r in result["ranges"]}, {"x", "dst"})
        for kid, nwarps in ((28, 1), (29, 2)):
            values = dict(ncols=32, ncols_dst_total=1, nchannels_dst=1, stride_row=32, stride_col_y=32,
                          stride_col_dst=32, stride_col_id=0, stride_row_id=0, channel_ratio=1, sample_ratio=1,
                          stride_channel_x=1024, stride_channel_y=128, stride_channel_dst=64,
                          stride_sample_x=1024, stride_sample_y=128, stride_sample_dst=64)
            c = call(kid, values, (1, 1, 1), (32, nwarps, 1), nwarps*16*36*4, ("ids",))
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"], result)
            # ncols_dst_total is not the native no-ID output guard: two columns
            # are materially written by this compiled 32x2 specialization.
            self.assertEqual(next(r["length_bytes"] for r in result["ranges"] if r["operand"] == "dst"), 256)

    def test_indirect_missing_contracts_never_infer_from_allocation(self):
        for kid in (10, 12, 20, 24, 25, 26, 27, 28, 30):
            c = call(kid, {}, (1, 1, 1))
            self.assertEqual(captured.evaluate_call(c, model())["reason"], "captured_indirect_or_scale_contract_required")

    def test_fixup_uses_own_abi_and_preserves_main_generation_requirement(self):
        values = dict(nrows_x=256, ncols_dst=1, stride_col_dst=256,
                      stride_channel_dst=256, stride_sample_dst=256)
        values.update({name: ranges.fastdiv_descriptor(size) for name, size in zip(
            ("blocks_per_ne00_fd", "nchannels_y_fd", "nsamples_y_fd", "ntx_fd"), (3, 1, 1, 1))})
        for kid in (24, 25, 26, 27):
            c = call(kid, values, (4, 4, 1), (32, 4, 1), 0, ("ids_dst", "expert_bounds"))
            rule, scalars = captured.direct_scalars(c)
            self.assertEqual(rule, "mmq_stream_k_fixup")
            self.assertFalse({"source_strides", "vector_strides", "vector_columns", "channel_ratio", "sample_ratio"} & scalars.keys())
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"], result)
            self.assertEqual({r["operand"] for r in result["ranges"]}, {"dst", "tmp_last_tile"})
            self.assertTrue(result["requires_matching_main_content_generation"])
            self.assertFalse(result["matching_main_content_generation_proven"])

    def test_copy_dimensions_derived_from_exact_elements_never_allocation(self):
        c = call(1, dict(ne=24, ne00=4, ne01=3, ne02=1, ne10=6, ne11=2, ne12=1,
                         nb00=4, nb01=16, nb02=48, nb03=48,
                         nb10=4, nb11=24, nb12=48, nb13=48), (1, 1, 1))
        rule, scalars = captured.direct_scalars(c)
        self.assertEqual(rule, "copy_f32")
        self.assertEqual(scalars["source_shape"], (4, 3, 1, 2))
        self.assertEqual(scalars["destination_shape"], (6, 2, 1, 2))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"])
        self.assertEqual([r["length_bytes"] for r in result["ranges"]], [96, 96])
        for field in c["fields"]:
            if field["value_type"] == "ptr":
                field["allocation_bytes"] = 20
        _, same_scalars = captured.direct_scalars(c)
        self.assertEqual(scalars, same_scalars)
        result = captured.evaluate_call(c, model())
        self.assertTrue(all(r["allocation_containment"] == "outside_observed_allocation" for r in result["ranges"]))
        self.assertFalse(result["memory_bounds_proven"])

    def test_copy_nondivisible_shape_remains_unresolved(self):
        c = call(1, dict(ne=25, ne00=4, ne01=3, ne02=1, ne10=5, ne11=5, ne12=1), (1, 1, 1))
        result = captured.evaluate_call(c, model())
        self.assertFalse(result["source_relative_ranges_derived"])
        self.assertEqual(result["reason"], "captured_dimension_not_exact")

    def test_convert_descriptor_and_mixed_widths(self):
        for kid, source_bytes, output_bytes in ((8, 2, 4), (9, 4, 2)):
            c = call(kid, dict(ne00=8, ne01=2, ne0203=6, ne02_fd=ranges.fastdiv_descriptor(3),
                               s01=8, s02=16, s03=48), (1, 2, 6))
            _, scalars = captured.direct_scalars(c)
            self.assertEqual(scalars["shape"], (8, 2, 3, 2))
            self.assertEqual((scalars["source_bytes"], scalars["destination_bytes"]), (source_bytes, output_bytes))
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"])

    def test_unary_k_is_element_count_n_is_row_width(self):
        c = call(22, dict(k=50, n=20, o0=24, o1=28), (2, 1, 1))
        _, scalars = captured.direct_scalars(c)
        self.assertEqual((scalars["elements"], scalars["row_columns"]), (50, 20))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"])
        self.assertEqual(next(r["length_bytes"] for r in result["ranges"] if r["operand"] == "dst"), 200)

    def test_broadcast_uses_pack_pointer_and_actual_fastdiv(self):
        values = dict(ne0=8, ne1=2, ne2=1, s00=1, s01=8, s02=16, s03=16,
                      s10=1, s11=8, s12=8, s13=8, s1=8, s2=16, s3=16)
        values.update({name: ranges.fastdiv_descriptor(size) for name, size in
                       zip(("ne3_fd", "ne10_fd", "ne11_fd", "ne12_fd", "ne13_fd"), (1, 8, 1, 1, 1))})
        c = call(3, values, (1, 2, 1))
        result = captured.evaluate_call(c, model())
        self.assertTrue(result["source_relative_ranges_derived"])
        operands = {r["operand"] for r in result["ranges"]}
        self.assertIn("src1s0", operands)
        self.assertNotIn("src1_unused", operands)

    def test_rms_actual_grid_and_pinned_template(self):
        for kid in (5, 6):
            values = dict(ncols=2048, stride_row=2048, stride_channel=4096, stride_sample=4096,
                          mul_stride_row=0, mul_stride_channel=0, mul_stride_sample=0)
            values.update({name: ranges.fastdiv_descriptor(size) for name, size in
                           zip(("mul_cols_fd", "mul_rows_fd", "mul_channels_fd", "mul_samples_fd"), (2048, 1, 1, 1))})
            c = call(kid, values, (2, 1, 1), (1024, 1, 1), 128)
            result = captured.evaluate_call(c, model())
            self.assertTrue(result["source_relative_ranges_derived"])
            self.assertEqual("mul" in {r["operand"] for r in result["ranges"]}, kid == 6)
            self.assertFalse(result["source_semantics_bound_to_cubin"])

    def test_bad_descriptor_and_geometry_fail_closed(self):
        c = call(8, dict(ne00=8, ne01=2, ne0203=6, ne02_fd=(1, 1, 3), s01=8, s02=16, s03=48), (1, 2, 6))
        result = captured.evaluate_call(c, model())
        self.assertEqual(result["reason"], "captured_fastdiv_invalid")
        c = call(22, dict(k=50, n=20, o0=24, o1=28), (1, 1, 1))
        result = captured.evaluate_call(c, model())
        self.assertEqual(result["reason"], "invalid_linear_geometry")
        self.assertEqual(result["ranges"], [])

    def test_indirect_family_is_not_derived_from_allocation(self):
        c = call(2, {}, (1, 1, 1))
        result = captured.evaluate_call(c, model())
        self.assertEqual(result["reason"], "captured_family_requires_additional_contract")
        self.assertFalse(result["source_relative_ranges_derived"])

    def test_unverified_model_is_not_accepted(self):
        c = call(22, dict(k=32, n=32, o0=32, o1=32), (1, 1, 1))
        with self.assertRaises(TypeError):
            captured.evaluate_call(c, object())


if __name__ == "__main__":
    unittest.main()
