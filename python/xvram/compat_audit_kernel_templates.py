"""Reviewed template choices keyed to the immutable 39-entry capture catalog.

These declarations describe the source specializations selected for audit. They
do not derive a source/binary identity edge, infer scalar dimensions, or certify
every future launch with a similar name. No runtime demangling is performed.
"""

from copy import deepcopy

from .compat_audit_memory import MemoryRuleError

CATALOG_SHA256 = "b8fd7a2fcf2aa61bb8de5ac7c85ea7f3e8b9aaff446fb899af3b7986af033494"
PROFILES = {
    1: {"rule": "copy_f32", "source_bytes": 4, "destination_bytes": 4},
    2: {"rule": "set_rows", "source_bytes": 4, "index_bytes": 8, "destination_bytes": 2},
    3: {"rule": "binary_broadcast", "operation": "add", "source_bytes": 4, "destination_bytes": 4, "pack_pointers": 1},
    4: {"rule": "binary_broadcast", "operation": "mul", "source_bytes": 4, "destination_bytes": 4, "pack_pointers": 1},
    5: {"rule": "rms_norm", "compiled_block_size": 1024, "multiply": False, "add": False},
    6: {"rule": "rms_norm", "compiled_block_size": 1024, "multiply": True, "add": False},
    7: {"rule": "softmax_256", "use_shared": True, "compiled_columns": 256, "compiled_block_size": 256, "mask_bytes": 4},
    8: {"rule": "convert_f16_f32", "source_bytes": 2, "destination_bytes": 4},
    9: {"rule": "convert_f16_f32", "source_bytes": 4, "destination_bytes": 2},
    10: {"rule": "matvec_f16", "source_bytes": 2, "accumulator": "f16", "output_columns": 1,
         "compiled_block_size": 128, "has_fusion": False, "is_multi_token_id": False},
    11: {"rule": "matvec_f16", "source_bytes": 2, "accumulator": "f32", "output_columns": 1,
         "compiled_block_size": 64, "has_fusion": False, "is_multi_token_id": False},
    12: {"rule": "matvec_quantized", "quant_type": 12, "output_columns": 1, "has_fusion": False, "small_k": False, "halve_iters": False},
    13: {"rule": "matvec_quantized", "quant_type": 12, "output_columns": 1, "has_fusion": True, "small_k": False, "halve_iters": False},
    14: {"rule": "matvec_quantized", "quant_type": 12, "output_columns": 2, "has_fusion": False, "small_k": False, "halve_iters": False},
    15: {"rule": "matvec_quantized", "quant_type": 14, "output_columns": 1, "has_fusion": False, "small_k": False, "halve_iters": False},
    16: {"rule": "matvec_quantized", "quant_type": 14, "output_columns": 1, "has_fusion": True, "small_k": False, "halve_iters": False},
    17: {"rule": "matvec_quantized", "quant_type": 14, "output_columns": 2, "has_fusion": False, "small_k": False, "halve_iters": False},
    18: {"rule": "quantize_q8_1_existing_SourceMemoryRules", "source_bytes": 4, "block_values": 32, "block_bytes": 36},
    19: {"rule": "get_rows_float", "source_bytes": 4, "index_bytes": 4, "destination_bytes": 4},
    20: {"rule": "quantize_mmq_q8_1", "ds_layout": 0, "scatter": False},
    21: {"rule": "quantize_mmq_q8_1", "ds_layout": 1, "scatter": False},
    22: {"rule": "unary_gated", "operation": "silu", "source_bytes": 4, "destination_bytes": 4},
    23: {"rule": "batched_pointer_tables", "pointer_bytes": 8},
    24: {"rule": "mmq_stream_k", "stage": "fixup", "quant_type": 12, "tile_columns": 128, "fallback": False},
    25: {"rule": "mmq_stream_k", "stage": "fixup", "quant_type": 12, "tile_columns": 40, "fallback": False},
    26: {"rule": "mmq_stream_k", "stage": "fixup", "quant_type": 14, "tile_columns": 128, "fallback": False},
    27: {"rule": "mmq_stream_k", "stage": "fixup", "quant_type": 14, "tile_columns": 40, "fallback": False},
    28: {"rule": "matf_half2", "source_type": "half2", "rows_per_block": 32, "cols_per_block": 2, "nwarps": 1, "has_ids": False},
    29: {"rule": "matf_half2", "source_type": "half2", "rows_per_block": 32, "cols_per_block": 2, "nwarps": 2, "has_ids": False},
    30: {"rule": "mmq_stream_k", "stage": "main", "quant_type": 12, "tile_columns": 128, "fallback": False},
    31: {"rule": "mmq_stream_k", "stage": "main", "quant_type": 12, "tile_columns": 40, "fallback": False},
    32: {"rule": "mmq_stream_k", "stage": "main", "quant_type": 14, "tile_columns": 128, "fallback": False},
    33: {"rule": "mmq_stream_k", "stage": "main", "quant_type": 14, "tile_columns": 40, "fallback": False},
    34: {"rule": "rope_neox", "forward": True, "has_freq_factors": False, "source_bytes": 4, "destination_bytes": 2},
    35: {"rule": "rope_neox", "forward": True, "has_freq_factors": False, "source_bytes": 4, "destination_bytes": 4},
}


def template_profile(catalog_id, catalog_sha256):
    if catalog_sha256 != CATALOG_SHA256:
        raise MemoryRuleError("template_catalog_identity_mismatch")
    if type(catalog_id) is not int or not 1 <= catalog_id <= 39:
        raise MemoryRuleError("unknown_template_catalog_id")
    if catalog_id >= 36:
        return {"catalog_id": catalog_id, "supported": False, "reason": "opaque_library_no_source_template_contract"}
    return {"catalog_id": catalog_id, "supported": True, "source_declaration_only": True,
            "binary_template_selection_proven": False, **deepcopy(PROFILES[catalog_id])}
