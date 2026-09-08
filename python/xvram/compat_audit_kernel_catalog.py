"""Generate a pinned typed capture catalog, without decoding native argument data.

The baseline evidence SHA identifies the exact reviewed 39 symbols. The source
family assignment below is reviewed, explicit, and never inferred from mangled
name fragments. Layout-size agreement alone is not a binary-source semantic
proof. Four opaque library kernels remain unsupported, not guessed structs.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .compat_audit_execution import _load, validate_report
from .compat_audit_kernel_ranges import SourceKernelRanges, capture_layout
from .compat_audit_memory import MemoryRuleError
from .compat_launch_probe import analyze_trace

BASELINE_SHA256 = "eed5de7453e99329cff64ab23849bc33391b25468c55c24fdf13ec3a9de8bb6c"
BASELINE_FAMILIES = (
    "cpy_scalar", "set_rows", "k_bin_bcast", "k_bin_bcast", "rms_norm_f32", "rms_norm_f32",
    "soft_max_f32", "convert_unary", "convert_unary", "mul_mat_vec_f", "mul_mat_vec_f",
    "mul_mat_vec_q", "mul_mat_vec_q", "mul_mat_vec_q", "mul_mat_vec_q", "mul_mat_vec_q", "mul_mat_vec_q",
    "quantize_q8_1", "get_rows_float", "quantize_mmq_q8_1", "quantize_mmq_q8_1", "unary_gated_op_kernel",
    "batched_pointer_tables", "mul_mat_q_stream_k_fixup", "mul_mat_q_stream_k_fixup",
    "mul_mat_q_stream_k_fixup", "mul_mat_q_stream_k_fixup", "mul_mat_f", "mul_mat_f",
    "mul_mat_q", "mul_mat_q", "mul_mat_q", "mul_mat_q", "rope_neox", "rope_neox",
    None, None, None, None,
)
TYPE_LAYOUT = {"ptr": (8, 8), "i64": (8, 8), "u64": (8, 8), "i32": (4, 4),
               "u32": (4, 4), "f32": (4, 4), "uint3": (12, 4), "f32x2": (8, 4),
               "bool": (1, 1), "fusion_struct": (48, 8), "soft_max_params_struct": (128, 8)}
STRUCT_FIELDS = {
    "fusion_struct": [("x_bias", "ptr", 0), ("gate", "ptr", 8), ("gate_bias", "ptr", 16),
                      ("x_scale", "ptr", 24), ("gate_scale", "ptr", 32),
                      ("glu_op", "i32", 40), ("glu_limit", "f32", 44)],
    "soft_max_params_struct": [("nheads", "i64", 0), ("n_head_log2", "u32", 8)] +
        [(name, "i64", 16+index*8) for index, name in enumerate(
            ("ncols", "nrows_x", "nrows_y", "ne00", "ne01", "ne02", "ne03", "nb11", "nb12", "nb13", "ne12", "ne13"))] +
        [(name, "f32", 112+index*4) for index, name in enumerate(("scale", "max_bias", "m0", "m1"))],
}


def expected_layout(family):
    arguments, cursor = [], 0
    for parameter in capture_layout(family)["arguments"]:
        kind = parameter["type"]
        size, alignment = TYPE_LAYOUT[kind]
        cursor = ((cursor+alignment-1)//alignment)*alignment
        argument = {**parameter, "offset_bytes": cursor, "size_bytes": size, "alignment_bytes": alignment,
                    "serialization": "resolved_allocation_reference" if kind == "ptr" else
                        "fieldwise_no_padding" if kind in STRUCT_FIELDS else "typed_scalar"}
        if kind in STRUCT_FIELDS:
            argument["fields"] = [{"name": name, "type": field_type, "offset_bytes": offset,
                                   "size_bytes": TYPE_LAYOUT[field_type][0],
                                   "serialization": "resolved_allocation_reference" if field_type == "ptr" else "typed_scalar"}
                                  for name, field_type, offset in STRUCT_FIELDS[kind]]
        arguments.append(argument)
        cursor += size
    return arguments


def generate(baseline_path, trace_path, source_root, extra_source_root):
    baseline, baseline_hash = _load(Path(baseline_path))
    if baseline_hash != BASELINE_SHA256:
        raise MemoryRuleError("unreviewed_kernel_catalog_baseline")
    validate_report(baseline)
    source_models = SourceKernelRanges(source_root, extra_source_root)
    trace = analyze_trace(Path(trace_path))
    if trace["sha256"] != baseline["provenance"]["probe_trace"]:
        raise MemoryRuleError("kernel_catalog_trace_hash_mismatch")
    layouts = {item["kernel_name"]: item for item in trace["layouts"]}
    if len(layouts) != len(trace["layouts"]) or len(layouts) != 39 or len(baseline["kernels"]) != 39:
        raise MemoryRuleError("kernel_catalog_layout_coverage")
    kernels = []
    for row, family in zip(baseline["kernels"], BASELINE_FAMILIES):
        name = row["name"]
        live = layouts.get(name)
        if live is None:
            raise MemoryRuleError("kernel_catalog_missing_live_layout")
        observed = [(p["offset_bytes"], p["size_bytes"]) for p in live["parameters"]]
        encoded = json.dumps(observed, separators=(",", ":")).encode("ascii")
        if hashlib.sha256(encoded).hexdigest() != row["layout_sha256"]:
            raise MemoryRuleError("kernel_catalog_layout_hash_mismatch")
        arguments = expected_layout(family) if family is not None else []
        if family is not None and [(p["offset_bytes"], p["size_bytes"]) for p in arguments] != observed:
            raise MemoryRuleError("kernel_catalog_source_abi_mismatch")
        kernels.append({"catalog_id": row["live_layout_id"], "symbol": name,
                        "source_family": family, "status": "typed_capture_candidate" if family else "unsupported_opaque_library",
                        "observed_layout_sha256": row["layout_sha256"], "observed_parameter_count": len(observed),
                        "static_candidate_cubin_sha256": [item["cubin_sha256"] for item in row["candidates"] if item["abi_match"]],
                        "arguments": arguments, "source_semantics_bound_to_cubin": False,
                        "memory_bounds_proven": False})
    return {"schema_version": 1, "catalog_type": "xvram.cuda_kernel_capture_catalog",
            "version": "0.1.0-dev", "profile": "llama-b10819-win-x64-qwen14b-observed39",
            "provenance": {"baseline_report_sha256": baseline_hash, "metadata_trace_sha256": trace["sha256"],
                           "source": source_models.provenance, "extra_sources": source_models.extra_provenance},
            "coverage": {"observed_layouts": 39, "typed_capture_candidates": 35, "opaque_unsupported": 4},
            "requirements": ["Native module/cubin edge must be independently bound for each invocation.",
                             "Reject argument-count/size/offset differences before capture.",
                             "Only fieldwise scalar values and resolved allocation IDs/offsets may be serialized.",
                             "Do not serialize pointer bytes, struct padding, pointer tables, or unknown opaque arguments.",
                             "Interior struct offsets are the source x64 ABI contract and require native static_assert checks.",
                             "Typed capture is not a memory-bounds, ordering, completeness, or oversubscription proof."],
            "kernels": kernels}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--extra-source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = generate(args.baseline, args.trace, args.source_root, args.extra_source_root)
    # A generator never overwrites evidence or a reviewed catalog implicitly.
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
