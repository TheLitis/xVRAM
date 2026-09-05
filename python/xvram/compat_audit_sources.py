"""Offline, conservative source-evidence index for the pinned native CUDA profile.

This is not a C++ parser, ABI verifier, tensor-bound inference tool or GO authority.
Only source bytes, reviewed source-only facts and candidate name correspondences
are established. The production runtime and frozen audit report are untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

COMMIT = "6a1a922d269908a29cbd4b49c27e6a8e7fd10fae"
BASE_URL = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{COMMIT}/"
BLOB_URL = f"https://github.com/ggml-org/llama.cpp/blob/{COMMIT}/"
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_SOURCES = 32
MAX_REPORTS = 16
MAX_KERNELS = 4096
MAX_U64 = (1 << 64) - 1
SOURCE_PREFIX = "ggml/src/ggml-cuda/"
_DEFINITION = re.compile(r"\b__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_$:.<>, ()*&\[\]-]{0,1023}\Z")
_HEX_ADDRESS = re.compile(r"0[xX][0-9a-fA-F]+")
_HASH = re.compile(r"[0-9a-f]{64}\Z")

# These obligations are a review queue, never argument layouts or pointer ranges.
FAMILY_OBLIGATIONS = {
    "cpy_scalar": ["strided_source_destination_extents", "aliasing_and_element_conversion"],
    "k_set_rows": ["device_index_values_and_destination_rows", "kv_alias_generation", "stride_units"],
    "k_bin_bcast": ["broadcast_dimensions_and_strides", "fused_pointer_arguments", "aliasing"],
    "rms_norm_f32": ["fused_scale_and_add_ranges", "row_channel_sample_strides"],
    "soft_max_f32": ["mask_and_position_bias_ranges", "soft_max_params_layout", "padded_columns"],
    "convert_unary": ["source_destination_dtype_and_strides", "valid_and_padded_extents"],
    "mul_mat_vec_f": ["fusion_struct_layout_and_pointers", "optional_indices", "matrix_strides_and_padding"],
    "mul_mat_vec_q": ["q4_k_q6_k_and_q8_1_layouts", "fusion_struct_layout_and_pointers", "optional_indices", "matrix_strides_and_padding"],
    "quantize_q8_1": ["valid_vs_padded_columns", "block_q8_1_output_extent", "row_channel_sample_strides"],
    "k_get_rows_float": ["device_index_values_and_source_rows", "strided_source_destination_extents"],
    "quantize_mmq_q8_1": ["mmq_q8_1_layout_variant", "scatter_indices_or_proven_null", "padded_output_extent"],
    "unary_gated_op_kernel": ["gate_pointer_and_offsets", "row_geometry", "aliasing"],
    "k_compute_batched_ptrs": ["device_pointer_table_contents", "indirect_cublas_operand_ranges", "table_consumer_ordering"],
    "mul_mat_q_stream_k_fixup": ["stream_k_scratch_extent", "destination_accumulation_ranges", "main_fixup_ordering"],
    "mul_mat_f": ["matrix_strides_and_padding", "optional_indices", "tile_tail_access"],
    "mul_mat_q": ["quant_layout_and_padding", "optional_ids_and_expert_bounds", "stream_k_scratch_extent", "tile_tail_access"],
    "rope_neox": ["position_and_frequency_ranges", "optional_set_rows_indices", "fp16_kv_alias_generation", "row_geometry"],
}

# Claims were manually reviewed against the pinned files. Every anchor and hash
# must verify before a claim is emitted as source-only proven. No captured launch
# is bound by these statements, even when its symbol resembles a definition.
SOURCE_FACTS = (
    {
        "id": "native_vmm_is_a_suballocated_arena",
        "claim": "The source reserves a 32 GiB VA arena, grows mapped physical segments, aligns pool allocations to 128 bytes and frees suballocations in LIFO order. VA capacity is not physical VRAM consumption.",
        "anchors": [("ggml-cuda.cu", "CUDA_POOL_VMM_MAX_SIZE = 1ull << 35"), ("ggml-cuda.cu", "const size_t alignment = 128;"), ("ggml-cuda.cu", "pool_used -= size;")],
        "obligation": "Correlate each reservation, physical segment, live suballocation and native pool generation; do not equate an allocation extent with a tensor range.",
    },
    {
        "id": "native_vmm_handle_release_is_not_unmap",
        "claim": "The source releases each physical handle after mapping and later unmaps the aggregate pool_size. Handle lifetime, mapping lifetime and reservation lifetime are distinct; native map and unmap call counts need not be equal.",
        "anchors": [("ggml-cuda.cu", "CU_CHECK(cuMemRelease(handle));"), ("ggml-cuda.cu", "CU_CHECK(cuMemUnmap(pool_addr, pool_size));"), ("ggml-cuda.cu", "CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));")],
        "obligation": "Prove physical coverage and complete unmap ranges rather than importing xVRAM's own per-frame call-count invariant into the native audit.",
    },
    {
        "id": "batched_gemm_has_gpu_pointer_indirection",
        "claim": "k_compute_batched_ptrs constructs source and destination pointer tables on GPU; the batched GEMM branch consumes those tables. A table allocation size does not establish the pointed-to operand bounds.",
        "anchors": [("ggml-cuda.cu", "static __global__ void k_compute_batched_ptrs("), ("ggml-cuda.cu", "ptrs_src[0*ne23 + i12 + i13*ne12] ="), ("ggml-cuda.cu", "alpha, (const void **) (ptrs_src.get() + 0*ne23)")],
        "obligation": "Bind table geometry, operand strides, table-writing launch, cuBLAS call and event order without guessing private cuBLAS kernel arguments.",
    },
    {
        "id": "fused_quantized_matvec_has_additional_pointers",
        "claim": "The fused matvec argument struct contains gate, bias and scale pointers. The quantized kernel conditionally reads fusion data and can select a channel through device indices. Matrix base pointers alone are insufficient.",
        "anchors": [("common.cuh", "struct ggml_cuda_mm_fusion_args_device {"), ("mmvq.cu", "channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]"), ("mmvq.cu", "use_gate      = fusion.gate      != nullptr;")],
        "obligation": "Verify compiled struct layout and all non-null pointer ranges, or prove optional inputs absent for every observed launch generation.",
    },
    {
        "id": "quantization_valid_and_padded_sizes_differ",
        "claim": "quantize_q8_1 distinguishes valid source columns from padded output columns and emits zero values for the padding. GGML's enum identifies Q4_K as 12 and Q6_K as 14; this explains source template values, not runtime argument bounds.",
        "anchors": [("quantize.cu", "const float xi = i0 < ne00 ?"), ("quantize.cu", "static __global__ void quantize_q8_1("), ("ggml/include/ggml.h", "GGML_TYPE_Q4_K    = 12,"), ("ggml/include/ggml.h", "GGML_TYPE_Q6_K    = 14,")],
        "obligation": "Prove the source strides, padded q8 layout and destination extent for each concrete launch; do not substitute GGUF tensor payload size.",
    },
    {
        "id": "mmq_stream_k_uses_global_fixup_scratch",
        "claim": "The source allocates a conditional stream-K fixup buffer in the native pool and launches a follow-up fixup kernel. Dynamic shared memory is not the whole scratch footprint.",
        "anchors": [("mmq.cuh", "tmp_fixup.alloc(block_nums_stream_k.x * config.J*config.I);"), ("mmq.cuh", "mul_mat_q_stream_k_fixup<type, J, fallback><<<")],
        "obligation": "Account for native pool bytes, both launches' accesses, event completion and pointer-table/workspace overlap before chunk-rounded admission.",
    },
)


class SourceIndexError(ValueError):
    """A bounded diagnostic code, deliberately excluding user file contents."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceIndexError("duplicate_json_key")
        result[key] = value
    return result


def _bounded_bytes(path, limit):
    try:
        with Path(path).open("rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise SourceIndexError("input_not_regular_file")
            data = stream.read(limit + 1)
    except OSError as error:
        raise SourceIndexError("input_unreadable") from error
    if len(data) > limit:
        raise SourceIndexError("input_size_limit")
    return data


def _json_bytes(data):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(SourceIndexError("non_finite_json")))
    except SourceIndexError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SourceIndexError("invalid_json") from error


def _uint(value):
    if type(value) is not int or not 0 <= value <= MAX_U64:
        raise SourceIndexError("invalid_counter")
    return value


def _add(a, b):
    value = a + b
    if value > MAX_U64:
        raise SourceIndexError("counter_overflow")
    return value


def load_profile():
    path = Path(__file__).with_name("compat_audit_source_profile.json")
    data = _bounded_bytes(path, MAX_SOURCE_BYTES)
    profile = _json_bytes(data)
    if not isinstance(profile, dict) or profile.get("upstream_commit") != COMMIT or profile.get("source_base_url") != BASE_URL or type(profile.get("schema_version")) is not int or profile.get("schema_version") != 1 or profile.get("profile_type") != "xvram.cuda_compat_source_profile":
        raise SourceIndexError("invalid_source_profile")
    entries = profile.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SOURCES:
        raise SourceIndexError("source_count_limit")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise SourceIndexError("invalid_source_entry")
        relative = entry["path"]
        if not isinstance(relative, str) or not re.fullmatch(r"ggml/(src|include)/[A-Za-z0-9_./-]+", relative) or ".." in relative or "//" in relative or str(PurePosixPath(relative)) != relative or relative in seen:
            raise SourceIndexError("invalid_source_path")
        if not isinstance(entry["sha256"], str) or not _HASH.fullmatch(entry["sha256"]) or not 0 < _uint(entry["size_bytes"]) <= MAX_SOURCE_BYTES:
            raise SourceIndexError("invalid_source_entry")
        seen.add(relative)
    return profile, hashlib.sha256(data).hexdigest()


def verify_sources(root, entries):
    root = Path(root).resolve()
    verified, result = {}, []
    for entry in sorted(entries, key=lambda item: item["path"]):
        relative = entry["path"]
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            path.resolve().relative_to(root)
            data = _bounded_bytes(path, MAX_SOURCE_BYTES)
            actual = hashlib.sha256(data).hexdigest()
            if len(data) != entry["size_bytes"] or actual != entry["sha256"]:
                status = "hash_mismatch"
            else:
                verified[relative] = data.decode("utf-8")
                status = "verified"
        except (ValueError, OSError, UnicodeError) as error:
            status = "unreadable" if isinstance(error, SourceIndexError) else "unsafe_or_invalid_source"
        result.append({"path": relative, "sha256": entry["sha256"], "size_bytes": entry["size_bytes"],
                       "status": status, "source_url": BASE_URL + relative})
    return verified, result


def source_definitions(verified):
    definitions = {}
    for path, source in sorted(verified.items()):
        for match in _DEFINITION.finditer(source):
            identifier = match.group(1)
            line = source.count("\n", 0, match.start()) + 1
            definitions.setdefault(identifier, []).append({"path": path, "line": line,
                                                           "source_url": BLOB_URL + path + f"#L{line}"})
    return definitions


def candidate_identifier(name):
    # A deliberately tiny Itanium top-level identifier recognizer, not demangling
    # or ABI parsing. Nested/library symbols stay opaque; template args stay intact.
    match = re.match(r"_Z([1-9][0-9]{0,2})", name)
    if not match:
        return None
    length = int(match.group(1))
    start = match.end()
    identifier = name[start:start + length]
    if len(identifier) != length or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        return None
    return identifier


def _fact_path(path):
    return path if path == "ggml/include/ggml.h" else SOURCE_PREFIX + path


def source_facts(verified):
    result = []
    for fact in SOURCE_FACTS:
        anchors, complete = [], True
        for filename, needle in fact["anchors"]:
            path = _fact_path(filename)
            source = verified.get(path, "")
            lines = [index + 1 for index, line in enumerate(source.splitlines()) if needle in line]
            if len(lines) != 1:
                complete = False
            else:
                anchors.append({"path": path, "line": lines[0], "source_url": BLOB_URL + path + f"#L{lines[0]}"})
        result.append({"fact_id": fact["id"], "status": "source_proven_only" if complete else "unverified_source",
                       "claim": fact["claim"] if complete else None, "anchors": anchors,
                       "captured_launch_binding": False, "remaining_obligation": fact["obligation"]})
    return result


def read_report(path):
    data = _bounded_bytes(path, MAX_REPORT_BYTES)
    report = _json_bytes(data)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("report_type") != "xvram.cuda_compat_audit":
        raise SourceIndexError("invalid_audit_report")
    provenance, coverage = report.get("provenance"), report.get("coverage")
    if not isinstance(provenance, dict) or provenance.get("upstream_commit") != COMMIT or not isinstance(coverage, dict):
        raise SourceIndexError("unpinned_or_invalid_audit_report")
    kernels = coverage.get("activity_kernel_types")
    if not isinstance(kernels, list) or len(kernels) > MAX_KERNELS:
        raise SourceIndexError("kernel_count_limit")
    counts, total = {}, 0
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise SourceIndexError("invalid_kernel_entry")
        name, count = kernel.get("name"), _uint(kernel.get("activities"))
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or _HEX_ADDRESS.search(name):
            raise SourceIndexError("invalid_or_private_kernel_name")
        if name in counts:
            raise SourceIndexError("duplicate_kernel_name")
        total = _add(total, count)
        counts[name] = count
    if total != _uint(coverage.get("kernel_activities")):
        raise SourceIndexError("activity_count_mismatch")
    # This validates the fields used by the index, not the entire frozen schema.
    return hashlib.sha256(data).hexdigest(), counts, total


def build_index(source_root, report_paths):
    if not 1 <= len(report_paths) <= MAX_REPORTS:
        raise SourceIndexError("report_count_limit")
    profile, profile_hash = load_profile()
    verified, sources = verify_sources(source_root, profile["files"])
    definitions = source_definitions(verified)
    reports, kernel_counts, by_report, total = [], Counter(), {}, 0
    for path in report_paths:
        digest, counts, count = read_report(path)
        if digest in by_report:
            raise SourceIndexError("duplicate_report")
        by_report[digest] = counts
        reports.append({"sha256": digest, "kernel_activities": count, "activity_kernel_types": len(counts)})
        total = _add(total, count)
        for name, value in counts.items():
            kernel_counts[name] = _add(kernel_counts[name], value)
        if len(kernel_counts) > MAX_KERNELS:
            raise SourceIndexError("kernel_count_limit")
    kernels, statuses = [], Counter()
    for name, count in sorted(kernel_counts.items()):
        identifier = candidate_identifier(name)
        candidates = definitions.get(identifier, [])
        status = "candidate_only" if len(candidates) == 1 else "ambiguous_source_candidates" if candidates else "unresolved"
        statuses[status] += 1
        obligations = FAMILY_OBLIGATIONS.get(identifier, ["opaque_or_unreviewed_kernel_semantics"])
        kernels.append({"name": name, "activities": count, "candidate_identifier": identifier,
                        "status": status, "candidate_definitions": candidates,
                        "observed_in": [{"report_sha256": digest, "activities": counts[name]} for digest, counts in sorted(by_report.items()) if name in counts],
                        "remaining_obligations": ["module_function_binary_identity", "compiled_argument_abi", *obligations, "live_ranges_and_event_order", "chunk_rounded_working_set_and_budget"],
                        "binary_binding_proven": False, "memory_bounds_proven": False})
    return {
        "schema_version": 1, "report_type": "xvram.cuda_compat_source_index",
        "provenance": {"upstream_commit": COMMIT, "source_profile_sha256": profile_hash,
                       "reports": sorted(reports, key=lambda item: item["sha256"])},
        "sources": sources, "source_facts": source_facts(verified), "kernels": kernels,
        "coverage": {"kernel_activities": total, "activity_kernel_types": len(kernels),
                     "candidate_only_types": statuses["candidate_only"], "ambiguous_types": statuses["ambiguous_source_candidates"],
                     "unresolved_types": statuses["unresolved"], "binary_bound_types": 0,
                     "complete_profile_coverage": False},
        "decision": {"verdict": "NO-GO", "interception_ready": False,
                     "reason": "Source-name candidates and reviewed source-only facts do not establish compiled ABI, captured tensor ranges, complete routing or lifecycle coverage."},
        "limitations": ["Input report fields are validated for this index; full frozen-schema validation is a separate gate.",
                        "GPU activity counts are not Runtime/Driver callback counts.",
                        "A source hash and matching identifier never bind a definition to an observed binary function.",
                        "Candidate matching is a review aid, not a C++ parser, demangler or memory contract verifier.",
                        "Absent families in CPU-offload captures do not prove absence from a future all-GPU profile."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--json", type=str, required=True, help="New output file, or '-' for stdout; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    try:
        result = build_index(args.source_root, args.report)
        output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except SourceIndexError as error:
        print(f"source index rejected: {error}", file=sys.stderr)
        return 23
    try:
        if args.json == "-":
            sys.stdout.write(output)
        else:
            with Path(args.json).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(output)
    except OSError:
        print("source index output failed", file=sys.stderr)
        return 74
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
