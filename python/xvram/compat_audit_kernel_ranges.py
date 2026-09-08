"""Pinned source arithmetic for indirect CUDA kernels; not a launch certificate.

No device addresses, CUDA calls, memory dereferences, or inferred tensor sizes.
Indices are explicit *declared* byte-offset/value snapshots. A native collector
must separately prove their content generation, producer retirement, lack of
intervening writes, argument ABI, cubin identity, and allocation/alias lifetimes.
The deliberately bounded profile rejects unsupported layouts rather than
substituting allocation extents for the actual source-level access set.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
from pathlib import Path

from . import compat_audit_sources as sources
from .compat_audit_memory import (
    I32_MAX, I64_MAX, U32_MAX, U64_MAX, MemoryRuleError,
    _geometry, _integer, _product, _sum,
)

MAX_ROWS = 100_000
MAX_INTERVALS = 400_000
SOURCE_LINES = {
    "ggml/src/ggml-cuda/common.cuh": [910, 927, 942],
    "ggml/src/ggml-cuda/set-rows.cu": [114, 154, 170, 177, 191, 213],
    "ggml/src/ggml-cuda/getrows.cu": [73, 87, 94, 98, 286],
    "ggml/src/ggml-cuda/rope.cu": [123, 147, 162, 171, 181, 203, 402, 436],
    "ggml/src/ggml-cuda/ggml-cuda.cu": [1341, 1363, 1581, 1595, 1608],
    "ggml/src/ggml-cuda/norm.cu": [77, 111, 136, 162, 319, 374],
    "ggml/src/ggml-cuda/quantize.cu": [54, 458, 493, 530, 575],
    "ggml/src/ggml-cuda/mmq.cuh": [18, 24, 27, 56],
    "ggml/src/ggml-cuda/mmvf.cu": [8, 46, 155, 379],
    "ggml/src/ggml-cuda/mmvq.cu": [77, 409, 536, 558, 655, 670, 733],
    "ggml/src/ggml-cuda/cpy.cu": [15, 27, 40, 215],
    "ggml/src/ggml-cuda/convert.cu": [417, 429, 437],
    "ggml/src/ggml-cuda/unary.cu": [263, 272, 276],
    "ggml/src/ggml-cuda/binbcast.cu": [36, 68, 81, 93, 98, 108],
    "ggml/src/ggml-cuda/softmax.cu": [25, 60, 74, 80, 105, 152],
    "ggml/src/ggml-cuda/mmf.cuh": [50, 111, 118, 166, 173, 198, 275, 657],
    "ggml/src/ggml-common.h": [258, 270, 338, 368],
}

# Ordinals are source declarations, NOT independently established cubin ABI.
# ptr means resolve to a lifetime-tagged ID/offset before serialization; never
# emit its 64-bit raw parameter bytes. uint3 fields may be emitted as integers.
SOURCE_ARGUMENTS = {
    "set_rows": tuple(zip(
        ("src0", "src1", "dst", "ne_total", "ne10", "ne11", "ne12", "ne13",
         "s01", "s02", "s03", "s10", "s11", "s12", "s1", "s2", "s3",
         "ne00_fd", "ne01_fd", "ne02_fd", "ne11_fd", "ne12_fd"),
        ("ptr",)*3 + ("i64",)*14 + ("uint3",)*5)),
    "get_rows_float": tuple(zip(
        ("src0", "src1", "dst", "ne00", "ne11", "ne12_fd", "s1", "s2", "s3",
         "nb01", "nb02", "nb03", "s10", "s11", "s12"),
        ("ptr",)*3 + ("i64",)*2 + ("uint3",) + ("u64",)*9)),
    "rope_neox": tuple(zip(
        ("x", "dst", "ne00", "ne01", "ne02", "s01", "s02", "s03", "s1", "s2", "s3",
         "n_dims", "n_offs", "pos", "freq_scale", "ext_factor", "attn_factor", "corr_dims",
         "theta_scale", "freq_factors", "row_indices", "set_rows_stride", "inplace"),
        ("ptr",)*2 + ("i32",)*11 + ("ptr",) + ("f32",)*3 + ("f32x2",) +
        ("f32",) + ("ptr",)*2 + ("i32", "bool"))),
    "quantize_q8_1": tuple(zip(
        ("x", "vy", "ne00", "s01", "s02", "s03", "ne0", "ne1", "ne2_fd"),
        ("ptr",)*2 + ("i64",)*5 + ("u32", "uint3"))),
    "quantize_mmq_q8_1": tuple(zip(
        ("x", "ids", "vy", "ne00", "s01", "s02", "s03", "ne0", "ne1", "ne2", "n_expert_used"),
        ("ptr",)*3 + ("i64",)*5 + ("i32",)*3)),
    "rms_norm_f32": tuple(zip(
        ("x", "dst", "ncols", "stride_row", "stride_channel", "stride_sample", "eps",
         "mul", "mul_stride_row", "mul_stride_channel", "mul_stride_sample", "mul_cols_fd",
         "mul_rows_fd", "mul_channels_fd", "mul_samples_fd", "add", "add_stride_row",
         "add_stride_channel", "add_stride_sample", "add_cols_fd", "add_rows_fd", "add_channels_fd", "add_samples_fd"),
        ("ptr",)*2 + ("i32",) + ("i64",)*3 + ("f32", "ptr") + ("i64",)*3 +
        ("uint3",)*4 + ("ptr",) + ("i64",)*3 + ("uint3",)*4)),
    "batched_pointer_tables": tuple(zip(
        ("src0", "src1", "dst", "ptrs_src", "ptrs_dst", "ne12", "ne13", "ne23", "nb02", "nb03",
         "nb12", "nb13", "nbd2", "nbd3", "r2", "r3"),
        ("ptr",)*5 + ("i64",)*3 + ("u64",)*6 + ("i64",)*2)),
    "mul_mat_q": tuple(zip(
        ("x", "y", "ids_dst", "expert_bounds", "dst", "tmp_fixup", "y_scale", "blocks_per_ne00_fd",
         "nrows_x", "ncols_dst", "stride_row_x", "ncols_y", "stride_col_dst", "channel_ratio_fd",
         "nchannels_y_fd", "stride_channel_x", "stride_channel_y", "stride_channel_dst", "sample_ratio_fd",
         "nsamples_y_fd", "stride_sample_x", "stride_sample_y", "stride_sample_dst", "ntx_fd"),
        ("ptr",)*7 + ("uint3",) + ("i32",)*5 + ("uint3",)*2 + ("i32",)*3 +
        ("uint3",)*2 + ("i32",)*3 + ("uint3",))),
    "mul_mat_q_stream_k_fixup": tuple(zip(
        ("ids_dst", "expert_bounds", "dst", "tmp_last_tile", "blocks_per_ne00_fd", "nrows_x", "ncols_dst",
         "stride_col_dst", "nchannels_y_fd", "stride_channel_dst", "nsamples_y_fd", "stride_sample_dst", "ntx_fd"),
        ("ptr",)*4 + ("uint3",) + ("i32",)*3 + ("uint3", "i32", "uint3", "i32", "uint3"))),
    "cpy_scalar": tuple(zip(
        ("cx", "cdst", "ne", "ne00", "ne01", "ne02", "nb00", "nb01", "nb02", "nb03",
         "ne10", "ne11", "ne12", "nb10", "nb11", "nb12", "nb13"), ("ptr",)*2 + ("i64",)*15)),
    "convert_unary": tuple(zip(
        ("vx", "y", "ne00", "ne01", "ne0203", "ne02_fd", "s01", "s02", "s03"),
        ("ptr",)*2 + ("i64",)*3 + ("uint3",) + ("i64",)*3)),
    "unary_gated_op_kernel": tuple(zip(("x", "g", "dst", "k", "n", "o0", "o1"),
                                       ("ptr",)*3 + ("i64",)*4)),
    "k_bin_bcast": tuple(zip(
        ("src0", "src1_unused", "dst", "ne0", "ne1", "ne2", "ne3_fd", "ne10_fd", "ne11_fd", "ne12_fd",
         "ne13_fd", "s1", "s2", "s3", "s00", "s01", "s02", "s03", "s10", "s11", "s12", "s13", "src1s0"),
        ("ptr",)*3 + ("u32",)*3 + ("uint3",)*5 + ("u32",)*11 + ("ptr",))),
    "soft_max_f32": tuple(zip(("x", "mask", "sinks", "dst", "params"),
                               ("ptr",)*4 + ("soft_max_params_struct",))),
    "mul_mat_f": tuple(zip(
        ("x", "y", "ids", "dst", "ncols", "ncols_dst_total", "nchannels_dst", "stride_row",
         "stride_col_y", "stride_col_dst", "stride_col_id", "stride_row_id", "channel_ratio", "stride_channel_x",
         "stride_channel_y", "stride_channel_dst", "sample_ratio", "stride_sample_x", "stride_sample_y", "stride_sample_dst"),
        ("ptr",)*4 + ("i32",)*16)),
}
_MATVEC_NAMES = ("x", "y", "ids", "fusion", "dst", "columns", "nchannels_y_fd", "stride_row_x",
                 "stride_col_y", "stride_col_dst", "channel_ratio_fd", "stride_channel_x",
                 "stride_channel_y", "stride_channel_dst", "sample_ratio_fd", "stride_sample_x",
                 "stride_sample_y", "stride_sample_dst", "ids_stride")
for _family, _type in (("mul_mat_vec_f", "i32"), ("mul_mat_vec_q", "u32")):
    SOURCE_ARGUMENTS[_family] = tuple(zip(_MATVEC_NAMES,
        ("ptr",)*3 + ("fusion_struct", "ptr", _type, "uint3") + (_type,)*3 +
        ("uint3",) + (_type,)*3 + ("uint3",) + (_type,)*4))


def capture_layout(family):
    """Minimum typed capture interface; a source ordinal never selects a kernel."""
    if not isinstance(family, str) or family not in SOURCE_ARGUMENTS:
        raise MemoryRuleError("unsupported_capture_family")
    return {"source_declaration_only": True, "binary_abi_proven": False,
            "arguments": [{"ordinal": i, "name": name, "type": kind}
                          for i, (name, kind) in enumerate(SOURCE_ARGUMENTS[family])],
            "fusion_members_in_source_order": ["x_bias", "gate", "gate_bias", "x_scale", "gate_scale", "glu_op", "glu_limit"]}

CAPTURE_REQUIREMENTS = {
    "all": [
        "call_id, context_id/generation, stream_id/generation, module SHA-256 and symbol/ABI hash",
        "grid/block/shared bytes and launch attributes including PDL; typed scalar arguments",
        "every pointer resolved to allocation_id/generation plus byte offset, never serialized VA",
        "live tensor/suballocation extents and read/write alias relations, not only allocation size",
        "fastdiv uint3 raw integer fields validated against the pinned initializer",
        "completion/order witness separate from arithmetic; no post-launch memory snapshot substitution",
    ],
    "indices": [
        "int32/int64 dtype, exact requested element byte offsets and signed values",
        "index allocation/content generation, last producer and completed observation boundary",
        "no mutation between witnessed content generation and consuming launch",
    ],
    "cublas": [
        "public API call_id, handle/context/stream generation, operation/dtype/compute/math modes",
        "m/n/k, lda/ldb/ldc, batchCount, pointer mode, finite alpha/beta values",
        "pointer tables linked to k_compute_batched_ptrs output generation and producer ordering",
        "exclusive API-to-private-kernel correlation and full library workspace/native allocation lifetime",
        "documented tensor access contract or separately established private-kernel bounds; ABI names alone are insufficient",
    ],
}


def _dims(values, count, maximum=I32_MAX):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise MemoryRuleError("invalid_dimension_vector")
    return tuple(_integer(v, maximum=maximum, positive=True) for v in values)


def _strides(values, count, maximum=I64_MAX):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise MemoryRuleError("invalid_stride_vector")
    return tuple(_integer(v, maximum=maximum) for v in values)


def _size(value):
    if type(value) is not int or value not in (2, 4):
        raise MemoryRuleError("unsupported_element_size")
    return value


def _rows(*dimensions):
    return _product(*dimensions, maximum=MAX_ROWS)


def _dot(indices, strides, maximum=I64_MAX):
    return _sum(*(_product(i, s, maximum=maximum) for i, s in zip(indices, strides)), maximum=maximum)


def fastdiv_descriptor(divisor):
    """Exact pinned initializer, within the no-overflow signed-32 subprofile."""
    divisor = _integer(divisor, maximum=I32_MAX, positive=True)
    shift = (divisor - 1).bit_length()
    multiplier = (((1 << 32) * ((1 << shift) - divisor)) // divisor + 1) & U32_MAX
    return multiplier, shift, divisor


def _descriptor(actual, divisor):
    if not isinstance(actual, (list, tuple)) or len(actual) != 3:
        raise MemoryRuleError("invalid_fastdiv_descriptor")
    actual = tuple(_integer(v, maximum=U32_MAX) for v in actual)
    if actual != fastdiv_descriptor(divisor):
        raise MemoryRuleError("fastdiv_descriptor_mismatch")


def _index_snapshot(value, width):
    if not isinstance(value, dict) or len(value) > MAX_ROWS:
        raise MemoryRuleError("invalid_index_snapshot")
    result = {}
    for offset, item in value.items():
        offset = _integer(offset)
        if offset % width:
            raise MemoryRuleError("unaligned_index_snapshot")
        # Negative device indices are not a legal nonnegative tensor subprofile.
        result[offset] = _integer(item, maximum=I32_MAX if width == 4 else I64_MAX)
    return result


class _Accesses:
    def __init__(self):
        self.items = {}
        self.count = 0

    def add(self, operand, mode, offset, length):
        offset, length = _integer(offset), _integer(length)
        end = _sum(offset, length)
        if not length:
            return
        self.count += 1
        if self.count > MAX_INTERVALS:
            raise MemoryRuleError("interval_count_limit")
        self.items.setdefault((operand, mode), []).append((offset, end))

    def result(self, rule, **extra):
        ranges = []
        for (operand, mode), intervals in sorted(self.items.items()):
            merged = []
            for first, end in sorted(intervals):
                if merged and first < merged[-1][1] and mode == "write":
                    raise MemoryRuleError("overlapping_destination_writes")
                if merged and first <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([first, end])
            ranges.extend({"operand": operand, "mode": mode, "offset_bytes": first,
                           "length_bytes": end - first} for first, end in merged)
        return {"model_version": 1, "rule": rule, "evidence_scope": "source_model_only",
                "runtime_binding_proven": False, "memory_bounds_proven": False,
                "device_ordering_proven": False, "admission_ready": False,
                "ranges": ranges, "global_scratch_bytes": 0,
                "requirements": deepcopy(CAPTURE_REQUIREMENTS["all"]), **extra}


def set_rows(*, shape, index_shape, source_strides, index_strides, destination_strides,
             descriptors, indices, grid, block, source_bytes=4, destination_bytes=2):
    """k_set_rows<float,int64,half>: strides are typed *elements*, not bytes.

    Return exact touched row intervals; repeated destination writes are rejected.
    index_shape is (ne10,ne11,ne12,ne13). ne13 is unused by this kernel.
    """
    n0, n1, n2, n3 = _dims(shape, 4)
    m0, m1, m2, _ = _dims(index_shape, 4)
    source_bytes, destination_bytes = _size(source_bytes), _size(destination_bytes)
    if (source_bytes, destination_bytes) != (4, 2):
        raise MemoryRuleError("outside_observed_set_rows_types")
    src, ind, dst = (_strides(v, 3) for v in (source_strides, index_strides, destination_strides))
    rows = _rows(n1, n2, n3)
    total = _product(n0, rows, maximum=I32_MAX)
    if m0 < n1:
        raise MemoryRuleError("index_tensor_too_short")
    grid, block = _geometry(grid), _geometry(block)
    if block[1:] != (1, 1) or grid[1:] != (1, 1) or block[0] > 1024:
        raise MemoryRuleError("invalid_set_rows_geometry")
    if grid[0] != (total + block[0] - 1) // block[0]:
        raise MemoryRuleError("set_rows_grid_mismatch")
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != 5:
        raise MemoryRuleError("invalid_fastdiv_descriptors")
    for actual, divisor in zip(descriptors, (n0, n1, n2, m1, m2)):
        _descriptor(actual, divisor)
    indices = _index_snapshot(indices, 8)
    accesses = _Accesses()
    for i3, i2, i1 in product(range(n3), range(n2), range(n1)):
        index_offset = _product(_dot((i1, i2 % m1, i3 % m2), ind), 8)
        if index_offset not in indices:
            raise MemoryRuleError("missing_index_value")
        row = indices[index_offset]
        src_start = _dot((i1, i2, i3), src)
        dst_start = _dot((row, i2, i3), dst)
        _sum(src_start, n0, maximum=I64_MAX)
        _sum(dst_start, n0, maximum=I64_MAX)
        accesses.add("src0", "read", _product(src_start, 4), _product(n0, 4))
        accesses.add("src1", "read", index_offset, 8)
        accesses.add("dst", "write", _product(dst_start, 2), _product(n0, 2))
    return accesses.result("set_rows_f32_i64_f16_v1", index_requirements=deepcopy(CAPTURE_REQUIREMENTS["indices"]))


def get_rows_float(*, columns, index_shape, source_byte_strides, index_strides,
                   destination_strides, descriptor, indices, grid, block,
                   source_bytes=4, destination_bytes=4):
    """k_get_rows_float<float,float>; source strides bytes, others elements."""
    columns = _integer(columns, maximum=I64_MAX, positive=True)
    m0, m1, m2 = _dims(index_shape, 3)
    _rows(m0, m1, m2)
    _product(m1, m2, maximum=I32_MAX)
    if (_size(source_bytes), _size(destination_bytes)) != (4, 4):
        raise MemoryRuleError("outside_observed_get_rows_types")
    src, ind, dst = (_strides(v, 3) for v in (source_byte_strides, index_strides, destination_strides))
    if any(s % 4 for s in src):
        raise MemoryRuleError("unaligned_source_stride")
    _descriptor(descriptor, m2)
    grid, block = _geometry(grid), _geometry(block)
    if grid[0] != m0 or block[1:] != (1, 1) or block[0] > 1024:
        raise MemoryRuleError("invalid_get_rows_geometry")
    _product(grid[1], block[0], maximum=I32_MAX)
    indices = _index_snapshot(indices, 4)
    accesses = _Accesses()
    for i12, i11, i10 in product(range(m2), range(m1), range(m0)):
        index_offset = _product(_dot((i10, i11, i12), ind), 4)
        if index_offset not in indices:
            raise MemoryRuleError("missing_index_value")
        src_start = _dot((indices[index_offset], i11, i12), src, maximum=U64_MAX)
        dst_start = _dot((i10, i11, i12), dst, maximum=U64_MAX)
        accesses.add("src0", "read", src_start, _product(columns, 4))
        accesses.add("src1", "read", index_offset, 4)
        accesses.add("dst", "write", _product(dst_start, 4), _product(columns, 4))
    return accesses.result("get_rows_f32_i32_f32_v1", index_requirements=deepcopy(CAPTURE_REQUIREMENTS["indices"]))


def rope_neox(*, shape, source_strides, destination_strides, n_dims, n_offs,
              grid, block, set_rows_stride=0, indices=None, inplace=False,
              source_bytes=4, destination_bytes=4, has_freq_factors=False):
    """Observed forward/no-frequency-factor rope; exact active column intervals.

    shape adds declared ne03 to the three native dimensions. The source has NO
    row_dst guard: launched rows must equal the independently bound tensor rows.
    Nonzero fused stride replaces the normal destination row calculation.
    """
    n0, n1, n2, n3 = _dims(shape, 4)
    rows = _rows(n1, n2, n3)
    _product(n1, n2, maximum=I32_MAX)
    src, dst = (_strides(v, 3, maximum=I32_MAX) for v in (source_strides, destination_strides))
    if _size(source_bytes) != 4:
        raise MemoryRuleError("outside_observed_rope_types")
    destination_bytes = _size(destination_bytes)
    n_dims = _integer(n_dims, maximum=I32_MAX, positive=True)
    n_offs = _integer(n_offs, maximum=I32_MAX)
    if n0 % 2 or n_dims % 2 or n_offs % 2 or _sum(n_offs, n_dims, maximum=I32_MAX) > n0:
        raise MemoryRuleError("invalid_rope_columns")
    if type(inplace) is not bool or has_freq_factors is not False:
        raise MemoryRuleError("outside_observed_rope_flags")
    set_rows_stride = _integer(set_rows_stride, maximum=I32_MAX)
    grid, block = _geometry(grid), _geometry(block)
    if (block[0] != 1 or block[2] != 1 or grid[2] != 1 or block[1] > 1024 or
            grid[0] != rows or grid[1] != (n0 + 2 * block[1] - 1) // (2 * block[1])):
        raise MemoryRuleError("invalid_rope_geometry")
    _product(2, block[1], grid[1], maximum=I32_MAX)
    if set_rows_stride:
        indices = _index_snapshot(indices, 8)
    elif indices is not None:
        raise MemoryRuleError("unexpected_unused_indices")
    accesses = _Accesses()
    first, length = (n_offs, n_dims) if inplace else (0, n0)
    for i3, i2, i1 in product(range(n3), range(n2), range(n1)):
        src_start = _dot((i1, i2, i3), src, maximum=I32_MAX)
        # Even the overwritten normal idst expression must not overflow first.
        dst_start = _dot((i1, i2, i3), dst, maximum=I32_MAX)
        _sum(dst_start, n0, maximum=I32_MAX)
        if set_rows_stride:
            if i2 * 8 not in indices:
                raise MemoryRuleError("missing_index_value")
            dst_start = _sum(_product(i1, dst[0], maximum=I32_MAX),
                             _product(indices[i2 * 8], set_rows_stride, maximum=I32_MAX), maximum=I32_MAX)
            accesses.add("row_indices", "read", i2 * 8, 8)
        _sum(src_start, n0, maximum=I32_MAX)
        _sum(dst_start, n0, maximum=I32_MAX)
        accesses.add("x", "read", _product(src_start + first, 4), _product(length, 4))
        accesses.add("dst", "write", _product(dst_start + first, destination_bytes), _product(length, destination_bytes))
        accesses.add("pos", "read", i2 * 4, 4)
    return accesses.result("rope_neox_forward_no_ff_v1", index_requirements=deepcopy(CAPTURE_REQUIREMENTS["indices"]),
                           position_values_affect_addresses=False, position_contents_needed_for_address_ranges=False)


def batched_pointer_tables(*, ne12, ne13, ne23, source0_byte_strides,
                           source1_byte_strides, destination_byte_strides,
                           broadcast_ratios, grid, block):
    """Exact generated pointer offsets, not memory accesses of the cuBLAS consumer.

    The kernel reads no pointed-to operand data. It only writes pointer tables.
    Relative targets must be resolved and joined to the consuming public API and
    table content generation before the opaque library kernels can be admitted.
    """
    ne12, ne13 = _dims((ne12, ne13), 2)
    count = _rows(ne12, ne13)
    if _integer(ne23, maximum=I64_MAX, positive=True) != count:
        raise MemoryRuleError("pointer_table_batch_count_mismatch")
    r2, r3 = _dims(broadcast_ratios, 2)
    s0, s1, sd = (_strides(v, 2, maximum=U64_MAX) for v in
                  (source0_byte_strides, source1_byte_strides, destination_byte_strides))
    grid, block = _geometry(grid), _geometry(block)
    if (block[2] != 1 or grid[2] != 1 or block[0] * block[1] > 1024 or
            grid[:2] != ((ne13 + block[0] - 1) // block[0], (ne12 + block[1] - 1) // block[1])):
        raise MemoryRuleError("invalid_pointer_table_geometry")
    _product(grid[0], block[0], maximum=U32_MAX)
    _product(grid[1], block[1], maximum=U32_MAX)
    targets = []
    for i13, i12 in product(range(ne13), range(ne12)):
        batch = i12 + i13 * ne12
        targets.append({"batch": batch,
                        "src0_offset_bytes": _dot((i12 // r2, i13 // r3), s0, maximum=U64_MAX),
                        "src1_offset_bytes": _dot((i12, i13), s1, maximum=U64_MAX),
                        "dst_offset_bytes": _dot((i12, i13), sd, maximum=U64_MAX)})
    accesses = _Accesses()
    accesses.add("ptrs_src", "write", 0, _product(count, 16))
    accesses.add("ptrs_dst", "write", 0, _product(count, 8))
    return accesses.result("batched_pointer_tables_v1", pointer_targets=targets,
                           consumer_memory_bounds_proven=False,
                           consumer_requirements=deepcopy(CAPTURE_REQUIREMENTS["cublas"]))


def rms_norm(*, columns, shape, source_strides, grid, block, shared_bytes,
             multiply=False, multiplier_shape=None, multiplier_strides=None,
             multiplier_descriptors=None):
    """Observed rms_norm_f32<1024,do_multiply,false>, nonnegative strides."""
    columns = _integer(columns, maximum=I32_MAX, positive=True)
    nrows, channels, samples = _dims(shape, 3)
    count = _rows(nrows, channels, samples)
    _product(count, columns, maximum=I32_MAX)
    _sum(columns, 1023, maximum=I32_MAX)
    src = _strides(source_strides, 3)
    grid, block = _geometry(grid), _geometry(block)
    if grid != (nrows, channels, samples) or block != (1024, 1, 1) or columns < 1024:
        raise MemoryRuleError("outside_observed_rms_geometry")
    if _integer(shared_bytes) != 128 or type(multiply) is not bool:
        raise MemoryRuleError("outside_observed_rms_configuration")
    if multiply:
        mul_dims = _dims(multiplier_shape, 4)
        mul_strides = _strides(multiplier_strides, 3)
        if not isinstance(multiplier_descriptors, (list, tuple)) or len(multiplier_descriptors) != 4:
            raise MemoryRuleError("invalid_fastdiv_descriptors")
        for actual, divisor in zip(multiplier_descriptors, mul_dims):
            _descriptor(actual, divisor)
    elif any(item is not None for item in (multiplier_shape, multiplier_strides, multiplier_descriptors)):
        raise MemoryRuleError("unexpected_unused_multiplier")
    accesses = _Accesses()
    for sample, channel, row in product(range(samples), range(channels), range(nrows)):
        start = _dot((row, channel, sample), src)
        _sum(start, columns, maximum=I64_MAX)
        accesses.add("x", "read", _product(start, 4), _product(columns, 4))
        linear_row = (sample * channels + channel) * nrows + row
        accesses.add("dst", "write", _product(linear_row, columns, 4), _product(columns, 4))
        if multiply:
            start = _dot((row % mul_dims[1], channel % mul_dims[2], sample % mul_dims[3]), mul_strides)
            length = min(columns, mul_dims[0])
            _sum(start, length, maximum=I64_MAX)
            accesses.add("mul", "read", _product(start, 4), _product(length, 4))
    return accesses.result("rms_norm_f32_1024_no_add_v1", dynamic_shared_bytes=128)


def quantize_mmq_q8_1(*, valid_columns, padded_columns, shape, source_strides,
                     ds_layout, grid, block, indices=None):
    """Non-scatter DS4/D4 kernels, including grid-padded channel spacing.

    The source loads float4, so valid columns and each base must be aligned to
    four elements. Unlike ordinary q8_1, its output is block-transposed.
    """
    valid = _integer(valid_columns, maximum=I64_MAX, positive=True)
    padded = _integer(padded_columns, maximum=I64_MAX, positive=True)
    rows, channels, samples = _dims(shape, 3)
    _rows(rows, channels, samples)
    src = _strides(source_strides, 3)
    if valid > padded or valid % 4 or padded % 128 or any(s % 4 for s in src):
        raise MemoryRuleError("invalid_mmq_quantize_shape")
    if type(ds_layout) is not int or ds_layout not in (0, 1):
        raise MemoryRuleError("outside_observed_mmq_ds_layout")
    grid, block = _geometry(grid), _geometry(block)
    if (block[1:] != (1, 1) or block[0] > 1024 or block[0] % 32 or
            grid != (rows, (padded + 4*block[0]-1)//(4*block[0]), channels*samples)):
        raise MemoryRuleError("invalid_mmq_quantize_geometry")
    # The native channel stride is based on launched lanes, not valid columns.
    pitch = _product(rows, grid[1], block[0], maximum=I64_MAX) // 32
    blocks = padded // 128
    if _product(rows, channels, samples, blocks) > MAX_INTERVALS // 2:
        raise MemoryRuleError("interval_count_limit")
    if indices is not None:
        indices = _index_snapshot(indices, 4)
    accesses = _Accesses()
    for sample, channel, row in product(range(samples), range(channels), range(rows)):
        selected_row = row
        if indices is not None:
            if row * 4 not in indices:
                raise MemoryRuleError("missing_index_value")
            selected_row = indices[row * 4]
            accesses.add("ids", "read", row * 4, 4)
        start = _dot((selected_row, channel, sample), src)
        _sum(start, valid, maximum=I64_MAX)
        accesses.add("x", "read", _product(start, 4), _product(valid, 4))
        base = _product(sample * channels + channel, pitch, maximum=I64_MAX)
        for k in range(blocks):
            ib = _sum(base, _product(k, rows, maximum=I64_MAX), row, maximum=I64_MAX)
            accesses.add("vy", "write", _product(ib, 144), 144)
    return accesses.result("quantize_mmq_q8_1_non_scatter_v1", output_block_bytes=144,
                           channel_pitch_blocks=pitch,
                           index_requirements=deepcopy(CAPTURE_REQUIREMENTS["indices"]))


def matvec_f16(*, columns2, shape, source_strides, vector_strides, destination_strides,
               channel_ratio, sample_ratio, descriptors, grid, block,
               compiled_block_size, shared_bytes, ids_null=True):
    """Observed half/half or half/float, ncols_dst=1, fusion=false, IDs absent.

    x strides are half elements, y/dst strides float elements. columns2 is native
    float2/half2 width, NOT scalar K. Positive row/channel products are narrowed
    to int before promotion in the source and must therefore fit signed-32.
    """
    columns2 = _integer(columns2, maximum=I32_MAX, positive=True)
    rows, channels, samples = _dims(shape, 3)
    _rows(rows, channels, samples)
    sx, sy, sd = (_strides(v, 3, maximum=I32_MAX) for v in
                  (source_strides, vector_strides, destination_strides))
    channel_ratio, sample_ratio = _dims((channel_ratio, sample_ratio), 2)
    if ids_null is not True:
        raise MemoryRuleError("outside_observed_matvec_null_ids_profile")
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != 2:
        raise MemoryRuleError("invalid_fastdiv_descriptors")
    _descriptor(descriptors[0], channel_ratio)
    _descriptor(descriptors[1], sample_ratio)
    if type(compiled_block_size) is not int or compiled_block_size not in (64, 128):
        raise MemoryRuleError("outside_observed_matvec_block")
    grid, block = _geometry(grid), _geometry(block)
    if grid != (rows, channels, samples) or block != (compiled_block_size, 1, 1):
        raise MemoryRuleError("invalid_matvec_geometry")
    if _integer(shared_bytes) != 128:
        raise MemoryRuleError("invalid_matvec_shared_bytes")
    _sum(columns2, compiled_block_size - 1, maximum=I32_MAX)
    accesses = _Accesses()
    for sample, channel, row in product(range(samples), range(channels), range(rows)):
        # The sample term explicitly promotes to i64, channel and row do not.
        x_start = _sum(_product(sample//sample_ratio, sx[2], maximum=I64_MAX),
                       _product(channel//channel_ratio, sx[1], maximum=I32_MAX),
                       _product(row, sx[0], maximum=I32_MAX), maximum=I64_MAX)
        y_start = _sum(_product(sample, sy[2], maximum=I64_MAX),
                       _product(channel, sy[1], maximum=I32_MAX), maximum=I64_MAX)
        d_start = _sum(_product(sample, sd[2], maximum=I64_MAX),
                       _product(channel, sd[1], maximum=I32_MAX), row, maximum=I64_MAX)
        if x_start % 2 or y_start % 2:
            raise MemoryRuleError("misaligned_matvec_vector_base")
        accesses.add("x", "read", _product(x_start, 2), _product(columns2, 4))
        accesses.add("y", "read", _product(y_start, 4), _product(columns2, 8))
        accesses.add("dst", "write", _product(d_start, 4), 4)
    return accesses.result("matvec_f16_f32_single_column_no_fusion_v1", dynamic_shared_bytes=128)


def matvec_quantized(*, quant_type, columns, output_rows, output_columns, channels, samples,
                     source_strides, vector_strides, destination_strides,
                     channel_ratio, sample_ratio, descriptors, grid, block,
                     compiled_arch, has_fusion=False, gate=False, bias=False,
                     gate_bias=False, ids_null=True):
    """Q4_K/Q6_K, sm_86 generic 4-warp profile; whole-block read bounds.

    These are conservative typed-block envelopes, not byte-exact transaction
    traffic. Padded x rows read by the 2-row kernel are INCLUDED even though its
    destination writes are guarded. x strides count Q4_K/Q6_K blocks, y strides
    count Q8_1 blocks, destination strides count float elements.
    """
    if type(quant_type) is not int or quant_type not in (12, 14):
        raise MemoryRuleError("outside_observed_matvec_quant_type")
    if type(compiled_arch) is not int or compiled_arch != 860 or ids_null is not True:
        raise MemoryRuleError("outside_observed_matvec_arch_or_ids")
    columns = _integer(columns, maximum=I32_MAX, positive=True)
    output_rows, channels, samples = _dims((output_rows, channels, samples), 3)
    if columns % 256 or type(output_columns) is not int or output_columns not in (1, 2):
        raise MemoryRuleError("invalid_quantized_matvec_shape")
    if any(type(v) is not bool for v in (has_fusion, gate, bias, gate_bias)):
        raise MemoryRuleError("invalid_fusion_flags")
    if (not has_fusion and any((gate, bias, gate_bias))) or (has_fusion and output_columns != 1):
        raise MemoryRuleError("outside_observed_quantized_fusion")
    sx, sy, sd = (_strides(v, 3, maximum=U32_MAX) for v in
                  (source_strides, vector_strides, destination_strides))
    if sd[0] != output_rows:
        raise MemoryRuleError("quantized_matvec_row_bound_mismatch")
    channel_ratio, sample_ratio = _dims((channel_ratio, sample_ratio), 2)
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != 2:
        raise MemoryRuleError("invalid_fastdiv_descriptors")
    _descriptor(descriptors[0], channel_ratio)
    _descriptor(descriptors[1], sample_ratio)
    rows_per_block = 1 if output_columns == 1 else 2
    grid, block = _geometry(grid), _geometry(block)
    if grid != ((output_rows + rows_per_block-1)//rows_per_block, channels, samples) or block != (32, 4, 1):
        raise MemoryRuleError("invalid_quantized_matvec_geometry")
    read_rows = grid[0] * rows_per_block
    kblocks = columns // 256
    block_bytes = 144 if quant_type == 12 else 210
    accesses = _Accesses()
    if (output_columns == 1 and channels == 1 and samples == 1 and not has_fusion
            and sx[0] == kblocks):
        # Exact union of adjacent whole-block row envelopes for the ordinary
        # vocabulary projection. No enumeration-cap increase and no clipping:
        # prove the same signed source-index intermediates at their maxima.
        x_last = _product(output_rows-1, sx[0], maximum=I32_MAX)
        x_blocks = _sum(x_last, kblocks, maximum=I32_MAX)
        _sum(columns//32, maximum=I32_MAX)
        _sum(output_rows-1, maximum=I32_MAX)
        accesses.add("x", "read", 0, _product(x_blocks, block_bytes))
        accesses.add("y", "read", 0, _product(columns//32, 36))
        accesses.add("dst", "write", 0, _product(output_rows, 4))
        return accesses.result("matvec_q4k_q6k_sm86_v1", read_bound_kind="whole_quantized_block_envelope",
            padded_source_rows=0, interval_strategy="analytical_contiguous_single_output",
            required_source_dependencies=["ggml/src/ggml-cuda/vecdotq.cuh"])
    _rows(read_rows, channels, samples)
    for sample, channel, row in product(range(samples), range(channels), range(read_rows)):
        x_start = _dot((row, channel//channel_ratio, sample//sample_ratio), sx, maximum=I32_MAX)
        _sum(x_start, kblocks, maximum=I32_MAX)
        accesses.add("x", "read", _product(x_start, block_bytes), _product(kblocks, block_bytes))
        if gate:
            accesses.add("gate", "read", _product(x_start, block_bytes), _product(kblocks, block_bytes))
        for col in range(output_columns):
            y_start = _dot((col, channel, sample), sy, maximum=I32_MAX)
            _sum(y_start, columns//32, maximum=I32_MAX)
            accesses.add("y", "read", _product(y_start, 36), _product(columns//32, 36))
            if row < output_rows:
                d_start = _sum(_dot((col, channel, sample), sd, maximum=I32_MAX), row, maximum=I32_MAX)
                accesses.add("dst", "write", _product(d_start, 4), 4)
                if bias:
                    accesses.add("x_bias", "read", _product(d_start, 4), 4)
                if gate and gate_bias:
                    accesses.add("gate_bias", "read", _product(d_start, 4), 4)
    return accesses.result("matvec_q4k_q6k_sm86_v1", read_bound_kind="whole_quantized_block_envelope",
                           padded_source_rows=read_rows-output_rows,
                           required_source_dependencies=["ggml/src/ggml-cuda/vecdotq.cuh"])


def mmq_stream_k(*, stage, quant_type, tile_columns, columns, output_rows, output_columns,
                 vector_columns, channels, samples, source_strides, vector_strides,
                 destination_strides, channel_ratio, sample_ratio, descriptors,
                 grid, block, shared_bytes, compiled_arch, ids_null=True):
    """sm86 Q4/Q6 main OR matching fixup launch; no IDs/experts/fallback.

    Simulate only integer stream-K partitioning, not GPU math. Include complete
    unguarded y tile loads (round_up(J*36,256) int32 values) and native fixup
    slots. This is still conditional source arithmetic with external lifetime,
    content-generation, workspace and binary proof obligations.
    """
    if stage not in ("main", "fixup"):
        raise MemoryRuleError("invalid_mmq_stage")
    if type(quant_type) is not int or quant_type not in (12, 14):
        raise MemoryRuleError("outside_observed_mmq_type")
    if type(tile_columns) is not int or tile_columns not in (40, 128):
        raise MemoryRuleError("outside_observed_mmq_tile")
    if type(compiled_arch) is not int or compiled_arch != 860 or ids_null is not True:
        raise MemoryRuleError("outside_observed_mmq_arch_or_ids")
    columns, output_rows, output_columns, vector_columns, channels, samples = _dims(
        (columns, output_rows, output_columns, vector_columns, channels, samples), 6)
    if columns % 256 or output_rows % 128 or output_columns > vector_columns:
        raise MemoryRuleError("outside_mmq_no_fallback_shape")
    sx, sy, sd = (_strides(v, n, maximum=I32_MAX) for v, n in
                  ((source_strides, 3), (vector_strides, 2), (destination_strides, 3)))
    if sd[0] < output_rows:
        raise MemoryRuleError("invalid_mmq_destination_stride")
    channel_ratio, sample_ratio = _dims((channel_ratio, sample_ratio), 2)
    blocks_k = columns // 256
    tiles_x = (output_columns + tile_columns - 1) // tile_columns
    tiles_y = output_rows // 128
    total = _product(blocks_k, tiles_x, tiles_y, channels, samples, maximum=(1 << 30)-1)
    if total > MAX_INTERVALS:
        raise MemoryRuleError("mmq_schedule_limit")
    divisors = (blocks_k, channel_ratio, channels, sample_ratio, samples, tiles_x)
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != 6:
        raise MemoryRuleError("invalid_fastdiv_descriptors")
    for actual, divisor in zip(descriptors, divisors):
        _descriptor(actual, divisor)
    grid, block = _geometry(grid), _geometry(block)
    nblocks = _integer(grid[0], maximum=MAX_ROWS, positive=True)
    load_ints = ((tile_columns * 36 + 255) // 256) * 256
    expected_shared = (tile_columns + load_ints + 128 * 76) * 4
    if stage == "main":
        if grid[1:] != (1, 1) or block != (32, 8, 1) or _integer(shared_bytes) != expected_shared:
            raise MemoryRuleError("invalid_mmq_main_geometry")
    elif grid[1:] != (4, 1) or block != (32, 4, 1) or _integer(shared_bytes) != 0:
        raise MemoryRuleError("invalid_mmq_fixup_geometry")
    starts = [i * total // nblocks for i in range(nblocks + 1)]
    partial_slots = {i for i in range(nblocks) if starts[i] < starts[i+1] and starts[i+1] % blocks_k}
    scratch_elements = _product(nblocks, tile_columns, 128, maximum=I32_MAX) if partial_slots else 0
    accesses = _Accesses()
    matrix_block_bytes = 144 if quant_type == 12 else 210

    def coordinates(tile):
        jt = tile % tiles_x
        tile //= tiles_x
        channel = tile % channels
        tile //= channels
        sample = tile % samples
        it = tile // samples
        if it >= tiles_y:
            raise MemoryRuleError("mmq_tile_outside_shape")
        return it, jt, channel, sample

    def output(tile, mode):
        it, jt, channel, sample = coordinates(tile)
        for col in range(jt*tile_columns, min((jt+1)*tile_columns, output_columns)):
            first = _sum(_dot((col, channel, sample), sd, maximum=I32_MAX), it*128, maximum=I32_MAX)
            _sum(first, 128, maximum=I32_MAX)
            accesses.add("dst", mode, _product(first, 4), 512)

    if stage == "main":
        for bid in range(nblocks):
            cursor, end = starts[bid], starts[bid+1]
            while cursor < end:
                tile, kfirst = divmod(cursor, blocks_k)
                stop = min(end, (tile+1)*blocks_k)
                klast = stop - tile*blocks_k
                it, jt, channel, sample = coordinates(tile)
                for row in range(it*128, (it+1)*128):
                    first = _sum(_dot((row, channel//channel_ratio, sample//sample_ratio), sx, maximum=I32_MAX),
                                 kfirst, maximum=I32_MAX)
                    _sum(first, klast-kfirst, maximum=I32_MAX)
                    accesses.add("x", "read", _product(first, matrix_block_bytes),
                                 _product(klast-kfirst, matrix_block_bytes))
                # The compiler must not be assumed to predicate tile tails.
                ybase = _sum(_dot((channel, sample), sy, maximum=I32_MAX),
                             jt*tile_columns*36, maximum=I32_MAX)
                for k in range(kfirst, klast):
                    for half in (0, 1):
                        first = _sum(ybase, _product(vector_columns, 2*k+half, 36, maximum=I32_MAX), maximum=I32_MAX)
                        _sum(first, load_ints, maximum=I32_MAX)
                        accesses.add("y", "read", _product(first, 4), _product(load_ints, 4))
                if stop % blocks_k:
                    accesses.add("tmp_fixup", "write", bid*tile_columns*128*4, tile_columns*128*4)
                else:
                    output(tile, "write")
                cursor = stop
    else:
        for bid in range(nblocks):
            first, end = starts[bid], starts[bid+1]
            if first == end or first % blocks_k == 0 or (first//blocks_k == end//blocks_k and end % blocks_k):
                continue
            previous, stop = bid-1, first
            while previous >= 0:
                cursor = starts[previous]
                if cursor == stop:
                    previous -= 1
                    stop = cursor
                    continue
                if previous not in partial_slots:
                    raise MemoryRuleError("mmq_fixup_reads_unwritten_slot")
                accesses.add("tmp_fixup", "read", previous*tile_columns*128*4, tile_columns*128*4)
                if cursor % blocks_k == 0 or cursor//blocks_k < first//blocks_k:
                    break
                previous -= 1
                stop = cursor
            else:
                raise MemoryRuleError("mmq_fixup_predecessor_underflow")
            output(first//blocks_k, "read")
            output(first//blocks_k, "write")
    return accesses.result("mmq_stream_k_sm86_no_ids_v1", stage=stage,
        global_scratch_bytes=_product(scratch_elements, 4),
        fixup_written_slots=sorted(partial_slots), y_tile_load_bytes=load_ints*4,
        required_dynamic_shared_bytes=expected_shared if stage == "main" else 0,
        requires_matching_main_content_generation=stage == "fixup",
        required_source_dependencies=["ggml/src/ggml-cuda/mmq-config-ampere.cuh",
                                      "ggml/src/ggml-cuda/mmq-load-tiles.cuh",
                                      "ggml/src/ggml-cuda/mmq-vec-dot.cuh",
                                      "ggml/src/ggml-cuda/mma.cuh"])


def mmq_stream_k_fixup(*, quant_type, tile_columns, blocks_per_row, output_rows, output_columns,
                      channels, samples, destination_strides, descriptors, grid, block,
                      shared_bytes, compiled_arch, ids_null=True):
    """No-ID sm86 fixup using only its own consumed ABI fields.

    The fixup has no x/y strides, channel broadcast ratios, or vector width.
    None are synthesized here. Scratch reads are conditional source ranges;
    matching main-kernel writes and their retired content generation remain
    independent mandatory obligations, not inferred from a previous call.
    """
    if type(quant_type) is not int or quant_type not in (12, 14):
        raise MemoryRuleError("outside_observed_mmq_type")
    if type(tile_columns) is not int or tile_columns not in (40, 128):
        raise MemoryRuleError("outside_observed_mmq_tile")
    if type(compiled_arch) is not int or compiled_arch != 860 or ids_null is not True:
        raise MemoryRuleError("outside_observed_mmq_arch_or_ids")
    blocks_k, output_rows, output_columns, channels, samples = _dims(
        (blocks_per_row, output_rows, output_columns, channels, samples), 5)
    if output_rows % 128:
        raise MemoryRuleError("outside_mmq_no_fallback_shape")
    sd = _strides(destination_strides, 3, maximum=I32_MAX)
    if sd[0] < output_rows:
        raise MemoryRuleError("invalid_mmq_destination_stride")
    tiles_x, tiles_y = (output_columns+tile_columns-1)//tile_columns, output_rows//128
    total = _product(samples, channels, tiles_x, tiles_y, blocks_k, maximum=(1 << 30)-1)
    if total > MAX_INTERVALS:
        raise MemoryRuleError("mmq_schedule_limit")
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != 4:
        raise MemoryRuleError("invalid_fastdiv_descriptors")
    for actual, divisor in zip(descriptors, (blocks_k, channels, samples, tiles_x)):
        _descriptor(actual, divisor)
    grid, block = _geometry(grid), _geometry(block)
    nblocks = _integer(grid[0], maximum=MAX_ROWS, positive=True)
    if grid[1:] != (4, 1) or block != (32, 4, 1) or _integer(shared_bytes) != 0:
        raise MemoryRuleError("invalid_mmq_fixup_geometry")
    # Pinned Ampere Q4_K/Q6_K config: I=128, K_vram=256, qk=256.
    # Thus native alignment by blocks_per_iter=K_vram/qk is exactly one.
    _product(nblocks, total, maximum=I64_MAX)
    scratch_elements = _product(nblocks, tile_columns, 128, maximum=I32_MAX)
    starts = [bid*total//nblocks for bid in range(nblocks+1)]
    partial_slots = {bid for bid in range(nblocks) if starts[bid] < starts[bid+1] and starts[bid+1] % blocks_k}
    accesses, read_slots = _Accesses(), set()
    for bid in range(nblocks):
        first, stop = starts[bid], starts[bid+1]
        if first == stop or first % blocks_k == 0 or (first//blocks_k == stop//blocks_k and stop % blocks_k):
            continue
        previous, boundary = bid-1, first
        while previous >= 0:
            cursor = starts[previous]
            if cursor == boundary:
                previous -= 1; boundary = cursor; continue
            if previous not in partial_slots:
                raise MemoryRuleError("mmq_fixup_reads_unwritten_slot")
            read_slots.add(previous)
            accesses.add("tmp_last_tile", "read", previous*tile_columns*128*4, tile_columns*128*4)
            if cursor % blocks_k == 0 or cursor//blocks_k < first//blocks_k:
                break
            previous -= 1; boundary = cursor
        else:
            raise MemoryRuleError("mmq_fixup_predecessor_underflow")
        coordinate = first//blocks_k
        coordinate, jt = divmod(coordinate, tiles_x)
        coordinate, channel = divmod(coordinate, channels)
        it, sample = divmod(coordinate, samples)
        if it >= tiles_y:
            raise MemoryRuleError("mmq_tile_outside_shape")
        offset = _sum(_product(sample, sd[2], maximum=I32_MAX),
                      _product(channel, sd[1], maximum=I32_MAX),
                      _product(jt, tile_columns, sd[0], maximum=I32_MAX),
                      _product(it, 128, maximum=I32_MAX), maximum=I32_MAX)
        for column in range(min(tile_columns, output_columns-jt*tile_columns)):
            first_element = _sum(offset, _product(column, sd[0], maximum=I32_MAX), maximum=I32_MAX)
            _sum(first_element, 127, maximum=I32_MAX)
            for mode in ("read", "write"):
                accesses.add("dst", mode, _product(first_element, 4), 128*4)
    return accesses.result("mmq_stream_k_fixup_sm86_no_ids_v1", stage="fixup",
        global_scratch_bytes=_product(scratch_elements, 4) if partial_slots else 0,
        expected_main_written_slots=sorted(partial_slots), required_scratch_slots=sorted(read_slots),
        requires_matching_main_content_generation=True, matching_main_content_generation_proven=False,
        required_source_dependencies=["ggml/src/ggml-cuda/mmq-config-ampere.cuh"])


def _linear_grid(elements, grid, block):
    grid, block = _geometry(grid), _geometry(block)
    if (grid[1:] != (1, 1) or block[1:] != (1, 1) or block[0] > 1024 or
            grid[0] != (elements + block[0] - 1)//block[0]):
        raise MemoryRuleError("invalid_linear_geometry")
    _sum(elements, block[0]-1, maximum=I64_MAX)


def copy_f32(*, elements, source_shape, destination_shape, source_byte_strides,
             destination_byte_strides, grid, block):
    """F32 cpy_scalar with contiguous innermost axis; reshape row boundaries."""
    source_shape, destination_shape = _dims(source_shape, 4), _dims(destination_shape, 4)
    elements = _integer(elements, maximum=I64_MAX, positive=True)
    if elements != _product(*source_shape, maximum=I64_MAX) or elements != _product(*destination_shape, maximum=I64_MAX):
        raise MemoryRuleError("copy_element_count_mismatch")
    s0, s1 = _strides(source_byte_strides, 4), _strides(destination_byte_strides, 4)
    if s0[0] != 4 or s1[0] != 4 or any(s % 4 for s in s0+s1):
        raise MemoryRuleError("outside_copy_contiguous_inner_axis")
    _linear_grid(elements, grid, block)
    accesses, cursor = _Accesses(), 0
    while cursor < elements:
        coords = []
        for shape in (source_shape, destination_shape):
            remaining = cursor
            indices = []
            for size in shape[:3]:
                remaining, coordinate = divmod(remaining, size)
                indices.append(coordinate)
            indices.append(remaining)
            coords.append(indices)
        count = min(source_shape[0]-coords[0][0], destination_shape[0]-coords[1][0])
        for operand, mode, indices, strides in (("cx", "read", coords[0], s0), ("cdst", "write", coords[1], s1)):
            first = _dot(indices, strides)
            _sum(first, count*4, maximum=I64_MAX)
            accesses.add(operand, mode, first, count*4)
        cursor += count
    return accesses.result("copy_scalar_f32_contiguous_inner_v1")


def convert_f16_f32(*, shape, source_strides, source_bytes, destination_bytes, descriptor, grid, block):
    """Observed convert_unary half→float or float→half; grid-stride rows."""
    n0, n1, n2, n3 = _dims(shape, 4)
    _rows(n1, n2, n3)
    _product(n2, n3, maximum=I32_MAX)
    if (_size(source_bytes), _size(destination_bytes)) not in ((2, 4), (4, 2)):
        raise MemoryRuleError("outside_observed_convert_types")
    src = _strides(source_strides, 3)
    _descriptor(descriptor, n2)
    grid, block = _geometry(grid), _geometry(block)
    if block[1:] != (1, 1) or block[0] > 1024 or grid != ((n0+block[0]-1)//block[0], min(n1, 65535), min(n2*n3, 65535)):
        raise MemoryRuleError("invalid_convert_geometry")
    total = _product(n0, n1, n2, n3, maximum=I64_MAX)
    accesses = _Accesses()
    for i3, i2, i1 in product(range(n3), range(n2), range(n1)):
        first = _dot((i1, i2, i3), src)
        _sum(first, n0, maximum=I64_MAX)
        accesses.add("vx", "read", _product(first, source_bytes), _product(n0, source_bytes))
    accesses.add("y", "write", 0, _product(total, destination_bytes))
    return accesses.result("convert_unary_f16_f32_v1")


def unary_gated(*, elements, row_columns, source_row_stride, gate_row_stride, grid, block):
    """Observed F32 SiLU gated kernel with optional final partial row."""
    elements, columns = _dims((elements, row_columns), 2, maximum=I64_MAX)
    s0, s1 = _strides((source_row_stride, gate_row_stride), 2)
    rows = (elements+columns-1)//columns
    _rows(rows)
    _linear_grid(elements, grid, block)
    accesses = _Accesses()
    for row in range(rows):
        count = min(columns, elements-row*columns)
        for operand, stride in (("x", s0), ("g", s1)):
            first = _product(row, stride, maximum=I64_MAX)
            _sum(first, count, maximum=I64_MAX)
            accesses.add(operand, "read", _product(first, 4), _product(count, 4))
    accesses.add("dst", "write", 0, _product(elements, 4))
    return accesses.result("unary_gated_silu_f32_v1")


def binary_broadcast(*, shape, broadcast_shape, source_strides, broadcast_strides,
                     destination_strides, descriptors, grid, block, source0_present=True):
    """Observed F32 add/mul with exactly ONE src1s pack pointer.

    src1 formal parameter is UNUSED in these observed specializations. Scalar
    stride holes are included in conservative read envelopes, not claimed read
    traffic; destination remains exact contiguous row intervals.
    """
    n0, n1, n2, n3 = _dims(shape, 4)
    m0, m1, m2, m3 = _dims(broadcast_shape, 4)
    _rows(n1, n2, n3)
    sx, sy, sd = (_strides(v, n, maximum=U32_MAX) for v, n in
                  ((source_strides, 4), (broadcast_strides, 4), (destination_strides, 3)))
    if type(source0_present) is not bool:
        raise MemoryRuleError("invalid_optional_pointer_flag")
    if not isinstance(descriptors, (list, tuple)) or len(descriptors) != 5:
        raise MemoryRuleError("invalid_fastdiv_descriptors")
    for actual, divisor in zip(descriptors, (n3, m0, m1, m2, m3)):
        _descriptor(actual, divisor)
    grid, block = _geometry(grid), _geometry(block)
    if _product(*block) > 1024:
        raise MemoryRuleError("invalid_broadcast_geometry")
    launched = tuple(_product(grid[i], block[i], maximum=I32_MAX) for i in range(3))
    if launched[1] < n1 or launched[2] < n2*n3:
        raise MemoryRuleError("incomplete_broadcast_geometry")
    accesses = _Accesses()
    for i3, i2, i1 in product(range(n3), range(n2), range(n1)):
        if source0_present:
            first = _dot((i1, i2, i3), sx[1:], maximum=U64_MAX)
            length = _sum(_product(n0-1, sx[0]), 1)
            accesses.add("src0", "read", _product(first, 4), _product(length, 4))
        first = _dot((i1%m1, i2%m2, i3%m3), sy[1:], maximum=U64_MAX)
        length = _sum(_product(min(n0, m0)-1, sy[0]), 1)
        accesses.add("src1s0", "read", _product(first, 4), _product(length, 4))
        first = _dot((i1, i2, i3), sd, maximum=U64_MAX)
        accesses.add("dst", "write", _product(first, 4), _product(n0, 4))
    return accesses.result("binary_broadcast_f32_single_pack_v1", read_bound_kind="affine_envelope_including_stride_holes")


def softmax_256(*, shape, mask_shape, mask_byte_strides, grid, block, shared_bytes,
                mask_present=True, sinks_present=False):
    """Observed soft_max_f32<true,256,256,float>; native params must bind shape.

    Template ncols=256 controls every load even if p.ncols says otherwise. Caller
    must bind ne00==256 independently; p.ncols is not used as a memory bound.
    """
    n0, n1, n2, n3 = _dims(shape, 4)
    m2, m3 = _dims(mask_shape, 2)
    _rows(n1, n2, n3)
    _product(n1, n2, n3, maximum=I32_MAX)
    sm = _strides(mask_byte_strides, 3)
    if n0 != 256 or any(s % 4 for s in sm):
        raise MemoryRuleError("outside_softmax_256_profile")
    if type(mask_present) is not bool or type(sinks_present) is not bool:
        raise MemoryRuleError("invalid_optional_pointer_flag")
    grid, block = _geometry(grid), _geometry(block)
    if grid != (n1, n2, n3) or block != (256, 1, 1) or _integer(shared_bytes) != (256+32)*4:
        raise MemoryRuleError("invalid_softmax_geometry")
    accesses = _Accesses()
    for i3, i2, i1 in product(range(n3), range(n2), range(n1)):
        first = ((i3*n2+i2)*n1+i1)*256*4
        accesses.add("x", "read", first, 1024)
        accesses.add("dst", "write", first, 1024)
        # The mask offset expression is evaluated even when mask is null.
        mask_offset = _dot((i1, i2%m2, i3%m3), sm)
        if mask_present:
            accesses.add("mask", "read", mask_offset, 1024)
        if sinks_present:
            accesses.add("sinks", "read", i2*4, 4)
    return accesses.result("softmax_shared_f32_256_v1", dynamic_shared_bytes=1152)


def matf_half2(*, columns2, output_rows, channels, samples, source_strides, vector_strides,
              destination_strides, channel_ratio, sample_ratio, grid, block,
              shared_bytes, compiled_arch, nwarps, ids_null=True):
    """Observed half2, rows_per_block=32, cols_per_block=2, no IDs.

    x strides and ncols count half2; y starts in F32 but stride_col_y counts
    float2; dst strides count F32. There are no row/column output guards in the
    non-ID branch: both columns and every launched row are material accesses.
    """
    columns2, output_rows, channels, samples = _dims((columns2, output_rows, channels, samples), 4)
    _rows(output_rows, channels, samples)
    if columns2 % 32 or output_rows % 32:
        raise MemoryRuleError("outside_matf_full_warp_and_row_tiles")
    if type(compiled_arch) is not int or compiled_arch != 860 or ids_null is not True:
        raise MemoryRuleError("outside_observed_matf_arch_or_ids")
    if type(nwarps) is not int or nwarps not in (1, 2):
        raise MemoryRuleError("outside_observed_matf_warps")
    sx, sy, sd = (_strides(v, 3, maximum=I32_MAX) for v in
                  (source_strides, vector_strides, destination_strides))
    channel_ratio, sample_ratio = _dims((channel_ratio, sample_ratio), 2)
    grid, block = _geometry(grid), _geometry(block)
    required_shared = max(nwarps*16*36*4, 8*(nwarps*32+4)*4)
    if grid != (output_rows//32, channels, samples) or block != (32, nwarps, 1) or _integer(shared_bytes) != required_shared:
        raise MemoryRuleError("invalid_matf_geometry")
    _sum(columns2, nwarps*32-1, maximum=I32_MAX)
    accesses = _Accesses()
    for sample, channel, row in product(range(samples), range(channels), range(output_rows)):
        x_first = _sum(_product(sample//sample_ratio, sx[2], maximum=I64_MAX),
                       _product(channel//channel_ratio, sx[1], maximum=I32_MAX),
                       _product(row, sx[0], maximum=I32_MAX), maximum=I64_MAX)
        _sum(_product(row % 32, sx[0], maximum=I32_MAX), columns2, maximum=I32_MAX)
        accesses.add("x", "read", _product(x_first, 4), _product(columns2, 4))
        for col in (0, 1):
            _sum(_product(col, sy[0], maximum=I32_MAX), columns2, maximum=I32_MAX)
            y_first = _sum(_product(sample, sy[2], maximum=I64_MAX),
                           _product(channel, sy[1], maximum=I32_MAX),
                           _product(col, sy[0], 2, maximum=I64_MAX), maximum=I64_MAX)
            if y_first % 2:
                raise MemoryRuleError("misaligned_matf_float2_base")
            accesses.add("y", "read", _product(y_first, 4), _product(columns2, 8))
            _sum(_product(col, sd[0], maximum=I32_MAX), row, maximum=I32_MAX)
            d_first = _sum(_product(sample, sd[2], maximum=I64_MAX),
                           _product(channel, sd[1], maximum=I32_MAX),
                           col*sd[0]+row, maximum=I64_MAX)
            accesses.add("dst", "write", _product(d_first, 4), 4)
    return accesses.result("matf_half2_32x2_sm86_v1", dynamic_shared_bytes=required_shared,
                           required_source_dependencies=["ggml/src/ggml-cuda/mma.cuh"])


def verify_extra_sources(source_root):
    """Verify the separately pinned reviewed subset without widening old profile."""
    path = Path(__file__).with_name("compat_audit_kernel_source_profile.json")
    try:
        data = sources._bounded_bytes(path, sources.MAX_SOURCE_BYTES)
        profile = sources._json_bytes(data)
        if (not isinstance(profile, dict) or profile.get("schema_version") != 1 or
                profile.get("profile_type") != "xvram.cuda_kernel_source_dependencies" or
                profile.get("upstream_commit") != sources.COMMIT or
                profile.get("source_base_url") != sources.BASE_URL):
            raise MemoryRuleError("invalid_kernel_source_dependency_profile")
        entries = profile.get("files")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
            raise MemoryRuleError("invalid_kernel_source_dependency_profile")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
                raise MemoryRuleError("invalid_kernel_source_dependency_entry")
            relative = entry["path"]
            if (not isinstance(relative, str) or not relative.startswith("ggml/src/ggml-cuda/") or
                    ".." in relative or "\\" in relative or "//" in relative or relative in seen or
                    not sources._HASH.fullmatch(entry["sha256"]) or
                    type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= sources.MAX_SOURCE_BYTES):
                raise MemoryRuleError("invalid_kernel_source_dependency_entry")
            seen.add(relative)
        _, results = sources.verify_sources(source_root, entries)
    except (sources.SourceIndexError, TypeError, KeyError) as error:
        raise MemoryRuleError("invalid_kernel_source_dependency_profile") from error
    if any(item["status"] != "verified" for item in results):
        raise MemoryRuleError("required_kernel_dependency_unverified")
    return {"profile_sha256": hashlib.sha256(data).hexdigest(), "upstream_commit": sources.COMMIT,
            "sources": results, "complete_build_dependency_closure": False}


class SourceKernelRanges:
    """Verify immutable pinned source bytes before issuing a source-only model."""

    def __init__(self, source_root, extra_source_root=None):
        try:
            profile, profile_hash = sources.load_profile()
            entries = [entry for entry in profile["files"] if entry["path"] in SOURCE_LINES]
            if len(entries) != len(SOURCE_LINES):
                raise MemoryRuleError("required_source_not_pinned")
            _, results = sources.verify_sources(source_root, entries)
        except sources.SourceIndexError as error:
            raise MemoryRuleError("source_profile_invalid") from error
        if any(item["status"] != "verified" for item in results):
            raise MemoryRuleError("required_source_unverified")
        self.provenance = {"upstream_commit": sources.COMMIT, "source_profile_sha256": profile_hash,
                           "sources": [{"path": item["path"], "sha256": item["sha256"],
                                        "reference_lines": SOURCE_LINES[item["path"]]} for item in results]}
        self.extra_provenance = verify_extra_sources(extra_source_root) if extra_source_root is not None else None

    def evaluate(self, rule, **scalars):
        rules = {"set_rows": set_rows, "get_rows_float": get_rows_float,
                 "rope_neox": rope_neox, "batched_pointer_tables": batched_pointer_tables,
                 "rms_norm": rms_norm, "quantize_mmq_q8_1": quantize_mmq_q8_1,
                 "matvec_f16": matvec_f16, "matvec_quantized": matvec_quantized,
                 "mmq_stream_k": mmq_stream_k, "mmq_stream_k_fixup": mmq_stream_k_fixup, "copy_f32": copy_f32,
                 "convert_f16_f32": convert_f16_f32, "unary_gated": unary_gated,
                 "binary_broadcast": binary_broadcast, "softmax_256": softmax_256,
                 "matf_half2": matf_half2}
        if not isinstance(rule, str) or rule not in rules:
            raise MemoryRuleError("unsupported_kernel_rule")
        if rule in ("matvec_quantized", "mmq_stream_k", "mmq_stream_k_fixup", "matf_half2", "copy_f32") and self.extra_provenance is None:
            raise MemoryRuleError("required_kernel_dependency_unverified")
        result = rules[rule](**scalars)
        result["source_provenance"] = deepcopy(self.provenance)
        if self.extra_provenance is not None:
            result["source_dependency_provenance"] = deepcopy(self.extra_provenance)
        return result
