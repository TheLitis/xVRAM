"""Bounded source-only memory arithmetic, never captured-pointer admission.

Rules consume explicitly supplied scalars. They do not decode launch arguments,
bind source to a binary, establish device ordering, or issue CUDA work. A verified
source model is not a verified invocation or an interoperability GO decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from . import compat_audit_sources as sources

U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1
U32_MAX = (1 << 32) - 1
I32_MAX = (1 << 31) - 1
MAX_RANGES = 100_000
PREFIX = "ggml/src/ggml-cuda/"
REQUIRED_SOURCES = {
    PREFIX + "quantize.cu": (54, 83, 94, 100, 558),
    "ggml/src/ggml-common.h": (89, 258, 270, 338, 368),
    PREFIX + "common.cuh": (910, 1176, 1200),
    PREFIX + "mmq.cuh": (1401, 1431, 1439, 1446),
    PREFIX + "ggml-cuda.cu": (571, 573, 664),
}


class MemoryRuleError(ValueError):
    """A stable diagnostic code without filesystem paths or native values."""


def _integer(value, *, maximum=U64_MAX, positive=False):
    if type(value) is not int or value < (1 if positive else 0) or value > maximum:
        raise MemoryRuleError("invalid_integer")
    return value


def _sum(*values, maximum=U64_MAX):
    result = sum(values)
    if result > maximum:
        raise MemoryRuleError("arithmetic_overflow")
    return result


def _product(*values, maximum=U64_MAX):
    result = 1
    for value in values:
        result *= value
        if result > maximum:
            raise MemoryRuleError("arithmetic_overflow")
    return result


def _ceil_div(value, divisor):
    # No unchecked value + divisor - 1, including for U64 range endpoints.
    return value // divisor + bool(value % divisor)


def _round_up(value, alignment):
    return _product(_ceil_div(value, alignment), alignment)


def _geometry(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise MemoryRuleError("invalid_launch_geometry")
    return tuple(_integer(item, maximum=U32_MAX, positive=True) for item in value)


def _verified_provenance(source_root):
    try:
        profile, profile_hash = sources.load_profile()
        entries = [entry for entry in profile["files"] if entry["path"] in REQUIRED_SOURCES]
        if len(entries) != len(REQUIRED_SOURCES):
            raise MemoryRuleError("required_source_not_pinned")
        _, results = sources.verify_sources(source_root, entries)
    except sources.SourceIndexError as error:
        raise MemoryRuleError("source_profile_invalid") from error
    if any(item["status"] != "verified" for item in results):
        raise MemoryRuleError("required_source_unverified")
    return {
        "upstream_commit": sources.COMMIT,
        "source_profile_sha256": profile_hash,
        "sources": [{"path": item["path"], "sha256": item["sha256"],
                     "reference_lines": list(REQUIRED_SOURCES[item["path"]])}
                    for item in results],
    }


def _result(rule, values):
    return {
        "model_version": 1,
        "rule": rule,
        "evidence_scope": "source_model_only",
        "runtime_binding_proven": False,
        "device_ordering_proven": False,
        "admission_ready": False,
        **values,
    }


def _quantize_q8_1(*, valid_columns, padded_columns, rows, channels, samples,
                    stride_row_elements, stride_channel_elements, stride_sample_elements,
                    grid, block, compiled_max_threads_per_block):
    """Source formula for the ordinary non-indexed q8 kernel under declared inputs.

    Selected bounded subprofile: positive dimensions; nonnegative F32 element
    strides (read aliases/broadcasts allowed); valid <= padded; padded % 32 == 0;
    a correctly initialized channels fastdiv descriptor; warp-aligned 1-D block;
    exact launcher grid; no more than 65,535 y/z blocks. The supplied compiled
    thread limit is a declaration, not a fact discovered by this function.
    """
    valid_columns = _integer(valid_columns, maximum=I64_MAX, positive=True)
    padded_columns = _integer(padded_columns, maximum=I64_MAX, positive=True)
    rows = _integer(rows, maximum=65_535, positive=True)
    channels = _integer(channels, maximum=65_535, positive=True)
    samples = _integer(samples, maximum=65_535, positive=True)
    strides = tuple(_integer(value, maximum=I64_MAX) for value in
                    (stride_row_elements, stride_channel_elements, stride_sample_elements))
    grid, block = _geometry(grid), _geometry(block)
    thread_limit = _integer(compiled_max_threads_per_block, maximum=1024, positive=True)
    if valid_columns > padded_columns or padded_columns % 32:
        raise MemoryRuleError("invalid_valid_or_padded_columns")
    if block[1:] != (1, 1) or block[0] % 32 or block[0] > thread_limit:
        raise MemoryRuleError("invalid_quantize_block")
    grid_z = _product(channels, samples, maximum=65_535)
    expected_grid = (_ceil_div(padded_columns, block[0]), rows, grid_z)
    if grid != expected_grid or grid[0] > I32_MAX:
        raise MemoryRuleError("quantize_grid_mismatch")
    # i_cont and each source element-index expression use signed int64_t.
    elements = _product(padded_columns, rows, channels, samples, maximum=I64_MAX)
    last_row_start = _sum(
        _product(rows - 1, strides[0], maximum=I64_MAX),
        _product(channels - 1, strides[1], maximum=I64_MAX),
        _product(samples - 1, strides[2], maximum=I64_MAX), maximum=I64_MAX)
    source_end_elements = _sum(last_row_start, valid_columns, maximum=I64_MAX)
    source_envelope = _product(source_end_elements, 4)
    write_bytes = _product(elements // 32, 36)
    return _result("quantize_q8_1_strided_f32_to_contiguous_q8_v1", {
        "read_envelopes": [{"operand": "x", "offset_bytes": 0, "length_bytes": source_envelope,
                            "includes_stride_holes": True}],
        "write_extents": [{"operand": "vy", "offset_bytes": 0, "length_bytes": write_bytes}],
        "valid_columns": valid_columns, "padded_columns": padded_columns,
        "row_count": _product(rows, channels, samples),
        "output_block_values": 32, "output_block_bytes": 36,
        "global_scratch_bytes": 0,
        "requirements": [
            "Scalar inputs, grid/block and compiled thread limit still need a binary-bound invocation proof.",
            "The uint3 channels descriptor must be produced by the pinned init_fastdiv_values(channels).",
            "x is F32; strides are elements, not bytes; vy is the pinned CUDA block_q8_1 layout.",
            "x and vy must not overlap; allocation lifetimes and the full extents remain to be bound.",
            "Read envelope includes stride holes and duplicate read aliases; it is not measured PCIe traffic.",
            "No CUDA_QUANTIZE_BLOCK_SIZE wrapper constant is inferred from the unpinned quantize.cuh.",
        ],
    })


def _stream_k_fixup(*, rows, columns_max, channels, samples, columns_x,
                    tile_rows, tile_columns, sm_count, declared_tiles, declared_blocks,
                    quant_type, stream_k_enabled, nvidia_profile):
    """The pinned NVIDIA host launcher's conditional float fixup allocation only.

    Tile/config/SM parameters are explicit declarations. This does not establish
    which compile-time MMQ configuration was selected or any x/y/dst access set.
    """
    rows, columns_max, channels, samples, columns_x = (
        _integer(value, maximum=I32_MAX, positive=True)
        for value in (rows, columns_max, channels, samples, columns_x))
    tile_rows = _integer(tile_rows, maximum=4096, positive=True)
    tile_columns = _integer(tile_columns, maximum=128, positive=True)
    sm_count = _integer(sm_count, maximum=I32_MAX, positive=True)
    declared_tiles = _integer(declared_tiles, maximum=I32_MAX, positive=True)
    declared_blocks = _integer(declared_blocks, maximum=U32_MAX, positive=True)
    if type(quant_type) is not int or quant_type not in (12, 14) or columns_x % 256:
        raise MemoryRuleError("unsupported_quantized_shape")
    if stream_k_enabled is not True or nvidia_profile is not True:
        raise MemoryRuleError("outside_stream_k_nvidia_profile")
    if tile_rows % 32 or tile_columns % 8:
        raise MemoryRuleError("unsupported_tile_configuration")
    # Device MMQ/fixup nrows_x + I - 1 is signed int, unlike mmq_args'
    # int64_t host dimensions. Keep both ceil numerators inside the deliberately
    # stricter shared 32-bit envelope; Python's safe ceil must not widen it.
    _sum(rows, tile_rows - 1, maximum=I32_MAX)
    _sum(columns_max, tile_columns - 1, maximum=I32_MAX)
    tiles_y = _ceil_div(rows, tile_rows)
    tiles_x = _ceil_div(columns_max, tile_columns)
    tiles = _product(tiles_x, tiles_y, channels, samples, maximum=I32_MAX)
    if declared_tiles != tiles:
        raise MemoryRuleError("declared_tile_count_mismatch")
    _sum(tiles, sm_count - 1, maximum=I32_MAX)
    waves = _ceil_div(tiles, sm_count)
    efficiency = (_product(100, tiles, maximum=I32_MAX) //
                  _product(sm_count, waves, maximum=I32_MAX))
    blocks = tiles if efficiency >= 90 else sm_count
    if declared_blocks != blocks:
        raise MemoryRuleError("declared_stream_k_blocks_mismatch")
    k_blocks = columns_x // 256
    if _product(tiles, k_blocks) >= 1 << 30:
        raise MemoryRuleError("stream_k_linear_index_limit")
    fixup_needed = tiles % blocks != 0
    # Host expression begins with uint32 grid.x; fixup device indices use int.
    # Enforce the stricter signed bound instead of accepting wrapping arithmetic.
    fixup_elements = _product(blocks, tile_rows, tile_columns, maximum=I32_MAX) if fixup_needed else 0
    payload = _product(fixup_elements, 4)
    return _result("nvidia_mmq_stream_k_fixup_allocation_v1", {
        "tile_count": tiles, "blocks": blocks, "tile_efficiency_percent": efficiency,
        "fixup_needed": fixup_needed, "scratch_payload_bytes": payload,
        "native_vmm_suballocation_bytes": _round_up(payload, 128),
        "native_vmm_physical_growth_bytes": None,
        "requirements": [
            "The actual type/J/fallback/config.I, stream-K selection and SM count still need binary-bound proof.",
            "Only the conditional global float fixup allocation is modelled; shared memory, MMQ operands and cuBLAS workspace are excluded.",
            "The native VMM pool must be the selected allocator for the 128-byte suballocation result.",
            "Physical growth also depends on the live pool's available space and VMM granularity; it is not this payload size.",
            "Main and fixup launches share this allocation until completion; lifetime and event ordering remain unproven.",
        ],
    })


class SourceMemoryRules:
    """Validate the pinned source set once, then evaluate declared scalar models."""

    def __init__(self, source_root):
        self._provenance = deepcopy(_verified_provenance(source_root))

    def quantize_q8_1(self, **scalars):
        result = _quantize_q8_1(**scalars)
        result["source_provenance"] = deepcopy(self._provenance)
        return result

    def stream_k_fixup(self, **scalars):
        result = _stream_k_fixup(**scalars)
        result["source_provenance"] = deepcopy(self._provenance)
        return result


@dataclass(frozen=True)
class AllocationRange:
    """Native IDs and relative offsets only. base_chunk_offset is base % chunk.

    IDs/generations are caller-supplied logical identities, never native handles.
    A base's modulo-chunk alignment is required, not silently assumed to be zero.
    """

    allocation_id: int
    generation: int
    allocation_bytes: int
    base_chunk_offset: int
    offset_bytes: int
    length_bytes: int


def chunk_rounded_union(ranges, *, chunk_bytes):
    """Union relative touched chunks within each proven allocation generation.

    Without absolute bases/shared-reservation alias proofs, separate allocations
    are counted separately. The sum is conservative, not exact physical residency.
    Zero-length ranges touch no chunks but must still be within the allocation.
    """
    chunk_bytes = _integer(chunk_bytes, positive=True)
    if not isinstance(ranges, (tuple, list)) or len(ranges) > MAX_RANGES:
        raise MemoryRuleError("range_count_limit")
    layouts, generations, grouped = {}, {}, {}
    for item in ranges:
        if not isinstance(item, AllocationRange):
            raise MemoryRuleError("invalid_allocation_range")
        allocation = _integer(item.allocation_id, positive=True)
        generation = _integer(item.generation, positive=True)
        size = _integer(item.allocation_bytes, positive=True)
        base = _integer(item.base_chunk_offset)
        offset = _integer(item.offset_bytes)
        length = _integer(item.length_bytes)
        if base >= chunk_bytes:
            raise MemoryRuleError("invalid_base_chunk_offset")
        end = _sum(offset, length)
        if end > size:
            raise MemoryRuleError("range_outside_allocation")
        previous_generation = generations.setdefault(allocation, generation)
        if previous_generation != generation:
            raise MemoryRuleError("ambiguous_allocation_generation")
        key = (allocation, generation)
        layout = layouts.setdefault(key, (size, base))
        if layout != (size, base):
            raise MemoryRuleError("inconsistent_allocation_layout")
        if length:
            first = _sum(base, offset) // chunk_bytes
            last = _ceil_div(_sum(base, end), chunk_bytes)
            grouped.setdefault(key, []).append((first, last))
    allocations, total_chunks = [], 0
    for key, intervals in sorted(grouped.items()):
        merged = []
        for first, last in sorted(intervals):
            if merged and first <= merged[-1][1]:
                merged[-1][1] = max(last, merged[-1][1])
            else:
                merged.append([first, last])
        count = 0
        for first, last in merged:
            count = _sum(count, last - first)
        total_chunks = _sum(total_chunks, count)
        allocations.append({"allocation_id": key[0], "generation": key[1],
                            "base_chunk_offset": layouts[key][1],
                            "relative_chunk_intervals": merged, "chunk_count": count})
    return _result("allocation_generation_chunk_union_v1", {
        "chunk_bytes": chunk_bytes, "allocations": allocations,
        "chunk_count_upper_bound": total_chunks,
        "rounded_bytes_upper_bound": _product(total_chunks, chunk_bytes),
        "exact_physical_residency_bytes": None,
        "requirements": [
            "IDs, generations, ranges and base modulo alignment are declarations until linked to the observed allocation lifetimes.",
            "Aliased/shared-reservation chunks across different allocation IDs are not merged; the total may overcount them.",
            "Chunk-rounded ranges do not prove that native reservations/mappings permit whole-chunk remapping.",
        ],
    })
