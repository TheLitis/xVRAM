"""Source-relative direct-kernel ranges from typed captured launch fields.

No shape comes from allocation size. Missing dimensions/indices and unsupported
templates remain unresolved. The verified source model does not establish that
the sampled cubin implements it; allocation containment is not a tensor bound.
"""
from collections import Counter
from copy import deepcopy
import math

from .compat_audit_kernel_arguments import CATALOG_SHA256, analyze_rows, bind_launches, BYTE_CAP, FIELD_CAP, CALL_CAP, RECORD_TYPE
from .compat_audit_identity import records, digest
from .compat_audit_kernel_ranges import SourceKernelRanges, fastdiv_descriptor, MAX_ROWS
from .compat_audit_kernel_templates import template_profile
from .compat_audit_memory import MemoryRuleError, _quantize_q8_1

DIRECT = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,
          23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}


def _positive(value):
    if type(value) is not int or not 1 <= value <= 2**63-1:
        raise MemoryRuleError("captured_dimension_invalid")
    return value


def _quotient(total, divisors):
    total = _positive(total)
    product = math.prod(_positive(item) for item in divisors)
    if product > 2**63-1 or total % product:
        raise MemoryRuleError("captured_dimension_not_exact")
    return _positive(total//product)


class Fields:
    def __init__(self, call):
        self.values = {(row["argument"], row["field"]): row for row in call["fields"]}
        if len(self.values) != len(call["fields"]):
            raise MemoryRuleError("captured_duplicate_field")

    def integer(self, name, field="value"):
        row = self.values.get((name, field))
        if row is None or row["value_type"] not in ("i32", "u32", "i64", "u64"):
            raise MemoryRuleError("captured_integer_missing")
        return row["value"]

    def integers(self, *names):
        return tuple(self.integer(name) for name in names)

    def boolean(self, name):
        row = self.values.get((name, "value"))
        if row is None or row["value_type"] != "bool" or type(row["value"]) is not bool:
            raise MemoryRuleError("captured_boolean_missing")
        return row["value"]

    def descriptor(self, name):
        rows = [self.values.get((name, field)) for field in ("x", "y", "z")]
        if any(row is None or row["value_type"] != "u32" for row in rows):
            raise MemoryRuleError("captured_fastdiv_missing")
        value = tuple(row["value"] for row in rows)
        if value != fastdiv_descriptor(value[2]):
            raise MemoryRuleError("captured_fastdiv_invalid")
        return value

    def pointer(self, name, field="value"):
        row = self.values.get((name, field))
        if row is None or row["value_type"] != "ptr":
            raise MemoryRuleError("captured_pointer_missing")
        return row

    def require_null(self, *names):
        if any(self.pointer(name)["resolution"] != "null" for name in names):
            raise MemoryRuleError("captured_indirect_or_scale_contract_required")

    def operand(self, name):
        if name in ("gate", "x_bias", "gate_bias"):
            return self.pointer("fusion", name)
        return self.pointer(name)


def direct_scalars(call):
    """Translate catalog-declared scalars only; pure unit-testable arithmetic."""
    kid = call["begin"]["catalog_id"]
    profile = template_profile(kid, CATALOG_SHA256)
    if kid not in DIRECT:
        raise MemoryRuleError("captured_family_requires_additional_contract")
    f, begin = Fields(call), call["begin"]
    common = {"grid": tuple(begin["grid_"+axis] for axis in "xyz"),
              "block": tuple(begin["block_"+axis] for axis in "xyz")}
    if kid in (10, 11, 12, 13, 14, 15, 16, 17):
        f.require_null("ids")
        # nchannels_y belongs only to the excluded ID branch. Pinned native
        # no-ID launches actually pass an unused zero descriptor here; do not
        # reinterpret that dead argument as a channel shape or divisor.
        descriptors = (f.descriptor("channel_ratio_fd"), f.descriptor("sample_ratio_fd"))
        scalars = dict(source_strides=f.integers("stride_row_x", "stride_channel_x", "stride_sample_x"),
                       vector_strides=f.integers("stride_col_y", "stride_channel_y", "stride_sample_y"),
                       destination_strides=f.integers("stride_col_dst", "stride_channel_dst", "stride_sample_dst"),
                       channel_ratio=descriptors[0][2], sample_ratio=descriptors[1][2],
                       descriptors=descriptors, ids_null=True)
        if kid in (10, 11):
            scalars.update(columns2=f.integer("columns"), shape=common["grid"],
                           compiled_block_size=profile["compiled_block_size"], shared_bytes=begin["shared_bytes"])
        else:
            # The native row guard is stride_col_dst, not allocation_bytes or
            # grid.x times an assumed tile height. Padded reads stay included.
            scalars.update(quant_type=profile["quant_type"], columns=f.integer("columns"),
                           output_rows=f.integer("stride_col_dst"), output_columns=profile["output_columns"],
                           channels=common["grid"][1], samples=common["grid"][2], compiled_arch=860,
                           has_fusion=profile["has_fusion"],
                           **{key: profile["has_fusion"] and f.pointer("fusion", member)["resolution"] != "null"
                              for key, member in (("gate", "gate"), ("bias", "x_bias"), ("gate_bias", "gate_bias"))})
        evaluated_rows = math.prod(_positive(value) for value in common["grid"])
        if kid in (14, 17):
            evaluated_rows *= 2
        analytical = (kid in (12, 15) and common["grid"][1:] == (1, 1)
                      and scalars["columns"] > 0 and scalars["columns"] % 256 == 0
                      and scalars["source_strides"][0] == scalars["columns"]//256)
        if evaluated_rows > MAX_ROWS and not analytical:
            # This is the bounded Python evaluator's enumeration limit, not
            # evidence of integer overflow or an unsafe native launch.
            raise MemoryRuleError("captured_source_evaluator_row_capacity")
    elif kid in (20, 21):
        f.require_null("ids")
        channels = _positive(f.integer("ne2"))
        scalars = dict(valid_columns=f.integer("ne00"), padded_columns=f.integer("ne0"),
                       shape=(f.integer("ne1"), channels, _quotient(common["grid"][2], (channels,))),
                       source_strides=f.integers("s01", "s02", "s03"), ds_layout=profile["ds_layout"])
    elif kid == 18:
        channels = f.descriptor("ne2_fd")[2]
        scalars = dict(valid_columns=f.integer("ne00"), padded_columns=f.integer("ne0"),
                       rows=f.integer("ne1"), channels=channels, samples=_quotient(common["grid"][2], (channels,)),
                       stride_row_elements=f.integer("s01"), stride_channel_elements=f.integer("s02"),
                       stride_sample_elements=f.integer("s03"), compiled_max_threads_per_block=1024)
    elif kid in (30, 31, 32, 33):
        f.require_null("ids_dst", "expert_bounds", "y_scale")
        descriptors = tuple(f.descriptor(name) for name in ("blocks_per_ne00_fd", "channel_ratio_fd",
                            "nchannels_y_fd", "sample_ratio_fd", "nsamples_y_fd", "ntx_fd"))
        scalars = dict(stage="main", quant_type=profile["quant_type"], tile_columns=profile["tile_columns"],
                       columns=descriptors[0][2]*256, output_rows=f.integer("nrows_x"),
                       output_columns=f.integer("ncols_dst"), vector_columns=f.integer("ncols_y"),
                       channels=descriptors[2][2], samples=descriptors[4][2],
                       source_strides=f.integers("stride_row_x", "stride_channel_x", "stride_sample_x"),
                       vector_strides=f.integers("stride_channel_y", "stride_sample_y"),
                       destination_strides=f.integers("stride_col_dst", "stride_channel_dst", "stride_sample_dst"),
                       channel_ratio=descriptors[1][2], sample_ratio=descriptors[3][2], descriptors=descriptors,
                       shared_bytes=begin["shared_bytes"], compiled_arch=860, ids_null=True)
    elif kid in (24, 25, 26, 27):
        f.require_null("ids_dst", "expert_bounds")
        descriptors = tuple(f.descriptor(name) for name in
                            ("blocks_per_ne00_fd", "nchannels_y_fd", "nsamples_y_fd", "ntx_fd"))
        scalars = dict(quant_type=profile["quant_type"], tile_columns=profile["tile_columns"],
                       blocks_per_row=descriptors[0][2], output_rows=f.integer("nrows_x"),
                       output_columns=f.integer("ncols_dst"), channels=descriptors[1][2], samples=descriptors[2][2],
                       destination_strides=f.integers("stride_col_dst", "stride_channel_dst", "stride_sample_dst"),
                       descriptors=descriptors, shared_bytes=begin["shared_bytes"], compiled_arch=860, ids_null=True)
    elif kid == 23:
        scalars = dict(ne12=f.integer("ne12"), ne13=f.integer("ne13"), ne23=f.integer("ne23"),
                       source0_byte_strides=f.integers("nb02", "nb03"),
                       source1_byte_strides=f.integers("nb12", "nb13"),
                       destination_byte_strides=f.integers("nbd2", "nbd3"), broadcast_ratios=f.integers("r2", "r3"))
    elif kid in (34, 35):
        if f.integer("set_rows_stride") != 0:
            # Distinct from pos: source rope.cu uses row_indices[i2] directly
            # in the destination address, so its content cannot be omitted.
            raise MemoryRuleError("captured_rope_row_indices_generation_required")
        ne00, ne01, ne02 = f.integers("ne00", "ne01", "ne02")
        scalars = dict(shape=(ne00, ne01, ne02, _quotient(common["grid"][0], (ne01, ne02))),
                       source_strides=f.integers("s01", "s02", "s03"),
                       destination_strides=f.integers("s1", "s2", "s3"), n_dims=f.integer("n_dims"),
                       n_offs=f.integer("n_offs"), inplace=f.boolean("inplace"),
                       source_bytes=profile["source_bytes"], destination_bytes=profile["destination_bytes"],
                       has_freq_factors=profile["has_freq_factors"], set_rows_stride=0)
    elif kid in (28, 29):
        f.require_null("ids")
        # ncols_dst_total only participates in the ID path; non-ID code writes
        # both compiled columns and every launched row with no output guard.
        scalars = dict(columns2=f.integer("ncols"), output_rows=common["grid"][0]*profile["rows_per_block"],
                       channels=common["grid"][1], samples=common["grid"][2],
                       source_strides=f.integers("stride_row", "stride_channel_x", "stride_sample_x"),
                       vector_strides=f.integers("stride_col_y", "stride_channel_y", "stride_sample_y"),
                       destination_strides=f.integers("stride_col_dst", "stride_channel_dst", "stride_sample_dst"),
                       channel_ratio=f.integer("channel_ratio"), sample_ratio=f.integer("sample_ratio"),
                       shared_bytes=begin["shared_bytes"], compiled_arch=860, nwarps=profile["nwarps"], ids_null=True)
    elif kid == 7:
        scalars = dict(shape=tuple(f.integer("params", name) for name in ("ne00", "ne01", "ne02", "ne03")),
                       mask_shape=tuple(f.integer("params", name) for name in ("ne12", "ne13")),
                       mask_byte_strides=tuple(f.integer("params", name) for name in ("nb11", "nb12", "nb13")),
                       shared_bytes=begin["shared_bytes"], mask_present=f.pointer("mask")["resolution"] != "null",
                       sinks_present=f.pointer("sinks")["resolution"] != "null")
    elif kid == 1:
        source = f.integers("ne00", "ne01", "ne02")
        destination = f.integers("ne10", "ne11", "ne12")
        elements = f.integer("ne")
        scalars = dict(elements=elements, source_shape=(*source, _quotient(elements, source)),
                       destination_shape=(*destination, _quotient(elements, destination)),
                       source_byte_strides=f.integers("nb00", "nb01", "nb02", "nb03"),
                       destination_byte_strides=f.integers("nb10", "nb11", "nb12", "nb13"))
    elif kid in (8, 9):
        descriptor = f.descriptor("ne02_fd")
        n2 = descriptor[2]
        scalars = dict(shape=(*f.integers("ne00", "ne01"), n2, _quotient(f.integer("ne0203"), (n2,))),
                       source_strides=f.integers("s01", "s02", "s03"), descriptor=descriptor,
                       source_bytes=profile["source_bytes"], destination_bytes=profile["destination_bytes"])
    elif kid == 22:
        # Source unary.cu: i >= k guard, j=(i/n)*stride+(i%n).
        scalars = dict(elements=f.integer("k"), row_columns=f.integer("n"),
                       source_row_stride=f.integer("o0"), gate_row_stride=f.integer("o1"))
    elif kid in (3, 4):
        descriptors = tuple(f.descriptor(name) for name in ("ne3_fd", "ne10_fd", "ne11_fd", "ne12_fd", "ne13_fd"))
        scalars = dict(shape=(*f.integers("ne0", "ne1", "ne2"), descriptors[0][2]),
                       broadcast_shape=tuple(value[2] for value in descriptors[1:]),
                       source_strides=f.integers("s00", "s01", "s02", "s03"),
                       broadcast_strides=f.integers("s10", "s11", "s12", "s13"),
                       destination_strides=f.integers("s1", "s2", "s3"), descriptors=descriptors,
                       source0_present=f.pointer("src0")["resolution"] != "null")
    else:
        scalars = dict(columns=f.integer("ncols"), shape=common["grid"],
                       source_strides=f.integers("stride_row", "stride_channel", "stride_sample"),
                       shared_bytes=begin["shared_bytes"], multiply=profile["multiply"])
        if profile["multiply"]:
            descriptors = tuple(f.descriptor(name) for name in ("mul_cols_fd", "mul_rows_fd", "mul_channels_fd", "mul_samples_fd"))
            scalars.update(multiplier_shape=tuple(value[2] for value in descriptors),
                           multiplier_strides=f.integers("mul_stride_row", "mul_stride_channel", "mul_stride_sample"),
                           multiplier_descriptors=descriptors)
    return ("mmq_stream_k_fixup" if kid in (24, 25, 26, 27) else profile["rule"]), dict(**scalars, **common)


def evaluate_call(call, source_model):
    if type(source_model) is not SourceKernelRanges:
        raise TypeError("captured_ranges_requires_verified_source_model")
    kid = call["begin"]["catalog_id"]
    result = {"catalog_id": kid, "status": "unresolved", "reason": None, "ranges": [],
              "source_relative_ranges_derived": False, "memory_bounds_proven": False,
              "source_semantics_bound_to_cubin": False, "alias_safety_proven": False}
    if call["status"] != "captured" or call["result"] != 0:
        result["reason"] = "capture_or_native_launch_not_successful"
        return result
    try:
        rule, scalars = direct_scalars(call)
        if rule == "quantize_q8_1_existing_SourceMemoryRules":
            # SourceKernelRanges has independently verified this same pinned
            # quantize.cu. Reuse the existing reviewed scalar arithmetic;
            # never infer source provenance from a caller-supplied filename.
            if not any(row["path"] == "ggml/src/ggml-cuda/quantize.cu"
                       for row in source_model.provenance["sources"]):
                raise MemoryRuleError("required_source_unverified")
            q8 = _quantize_q8_1(**scalars)
            relative = dict(source_provenance=deepcopy(source_model.provenance), ranges=[
                dict(operand=r["operand"], mode=mode, offset_bytes=r["offset_bytes"], length_bytes=r["length_bytes"])
                for key, mode in (("read_envelopes", "read"), ("write_extents", "write")) for r in q8[key]])
        else:
            relative = source_model.evaluate(rule, **scalars)
        result.update(status="source_relative_only", reason="source_binary_and_tensor_contracts_unresolved",
                      source_relative_ranges_derived=True, rule=rule,
                      source_provenance=relative["source_provenance"])
        if "source_dependency_provenance" in relative:
            result["source_dependency_provenance"] = relative["source_dependency_provenance"]
        for requirement in ("requires_matching_main_content_generation", "matching_main_content_generation_proven",
                            "required_scratch_slots", "expected_main_written_slots", "interval_strategy",
                            "position_values_affect_addresses", "position_contents_needed_for_address_ranges",
                            "consumer_memory_bounds_proven", "consumer_requirements"):
            if requirement in relative:
                result[requirement] = relative[requirement]
        f = Fields(call)
        for access in relative["ranges"]:
            pointer = f.operand(access["operand"])
            bound = dict(access, pointer_resolution=pointer["resolution"], allocation_containment="unresolved")
            if pointer["resolution"] == "allocation_identity":
                begin = pointer["offset_bytes"]+access["offset_bytes"]
                end = begin+access["length_bytes"]
                bound.update(allocation_id=pointer["allocation_id"], generation=pointer["generation"],
                             memory_revision=pointer["memory_revision"], memory_sequence=pointer["memory_sequence"],
                             allocation_offset_bytes=begin,
                             allocation_containment="inside_observed_allocation" if end <= pointer["allocation_bytes"] and end < 2**64 else "outside_observed_allocation")
            elif pointer["resolution"] == "null":
                bound["allocation_containment"] = "required_pointer_null"
            result["ranges"].append(bound)
        if "pointer_targets" in relative:
            result["symbolic_pointer_writes"] = []
            result["table_content_generation_proven"] = False
            result["consumer_memory_bounds_proven"] = False
            count = scalars["ne23"]
            for target in relative["pointer_targets"]:
                batch = target["batch"]
                for base, table, index, offset_key in (("src0", "ptrs_src", batch, "src0_offset_bytes"),
                        ("src1", "ptrs_src", count+batch, "src1_offset_bytes"),
                        ("dst", "ptrs_dst", batch, "dst_offset_bytes")):
                    pointer, table_pointer = f.pointer(base), f.pointer(table)
                    symbolic = dict(batch=batch, table_operand=table, table_offset_bytes=index*8,
                                    element_bytes=8, target_operand=base, target_offset_bytes=target[offset_key],
                                    target_resolution=pointer["resolution"], table_resolution=table_pointer["resolution"],
                                    table_content_generation_proven=False, consumer_memory_bounds_proven=False,
                                    pointee_bytes_accessed_by_producer=0, target_containment="unresolved")
                    if pointer["resolution"] == "allocation_identity":
                        target_offset = pointer["offset_bytes"] + target[offset_key]
                        symbolic.update(target_allocation_id=pointer["allocation_id"], target_generation=pointer["generation"],
                            target_allocation_offset_bytes=target_offset, memory_revision=pointer["memory_revision"],
                            memory_sequence=pointer["memory_sequence"],
                            target_containment="inside_or_one_past_observed_allocation"
                            if target_offset <= pointer["allocation_bytes"] and target_offset < 2**64
                            else "outside_observed_allocation")
                    if table_pointer["resolution"] == "allocation_identity":
                        symbolic.update(table_allocation_id=table_pointer["allocation_id"],
                            table_generation=table_pointer["generation"],
                            table_allocation_offset_bytes=table_pointer["offset_bytes"]+index*8,
                            table_memory_revision=table_pointer["memory_revision"],
                            table_memory_sequence=table_pointer["memory_sequence"])
                    result["symbolic_pointer_writes"].append(symbolic)
    except MemoryRuleError as error:
        # Preserve the exact reviewed rule rejection, never use an allocation
        # extent or an optimistic alternate rule to fill a missing dimension.
        result.update(status="unresolved", reason=str(error), source_relative_ranges_derived=False, ranges=[])
    return result


def analyze(path, launch_path, source_model):
    before = digest(path, BYTE_CAP)
    counts, calls = analyze_rows(records(path, RECORD_TYPE, cap=BYTE_CAP, max_records=FIELD_CAP+3*CALL_CAP+2))
    launch_sha = bind_launches(calls, launch_path)
    derived = Counter()
    rejected = Counter()
    containment = Counter()
    symbolic_containment = Counter()
    symbolic_writes = table_generations = 0
    for call in calls.values():
        result = evaluate_call(call, source_model)
        if result["source_relative_ranges_derived"]:
            derived[str(result["catalog_id"])] += 1
            containment.update(row["allocation_containment"] for row in result["ranges"])
            if "symbolic_pointer_writes" in result:
                symbolic_writes += len(result["symbolic_pointer_writes"])
                table_generations += 1
                symbolic_containment.update(row["target_containment"] for row in result["symbolic_pointer_writes"])
        else:
            rejected[result["reason"]] += 1
    if before != digest(path, BYTE_CAP):
        raise ValueError("captured_range_inputs_changed")
    return {"catalog_sha256": CATALOG_SHA256, "argument_trace_sha256": before,
            "launch_trace_sha256": launch_sha, "calls": counts["calls"],
            "source_relative_calls_by_catalog": dict(derived), "unresolved_calls_by_reason": dict(rejected),
            "allocation_containment_observations": dict(containment),
            "symbolic_pointer_writes": symbolic_writes, "pointer_table_generations_unproven": table_generations,
            "symbolic_target_containment_observations": dict(symbolic_containment),
            "memory_bounds_proven": False, "native_launches_cubin_bound": False,
            "stream_event_order_proven": False, "trace_complete": False}
