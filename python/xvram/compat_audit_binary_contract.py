"""Strict, dependency-free contract for observation-only cubin evidence v1.

Validation does not turn static parameter layouts into runtime argument or
memory-bound proofs. Error messages are fixed codes: untrusted names, paths,
addresses and diagnostic output are never interpolated into them.
"""
from __future__ import annotations

from collections import Counter
import json
import re

from .compat_audit_sources import COMMIT, _HEX_ADDRESS, _SAFE_NAME

SCHEMA_VERSION = 1
REPORT_TYPE = "xvram.cuda_compat_binary_evidence"
BACKEND_SHA256 = "88350839e27a43212a52cf6686562b2b6ef1498c59391dfb8cda49f3ebd89a62"
TOOL_SHA256 = "cb72e7e353ed20320deb4770f75291a828c0b23ca8c2d27688c4eeced96543f5"
TOOL_VERSION = "13.3.73"
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
MAX_U64 = (1 << 64) - 1
LIMITATIONS = (
    "static_symbol_matches_are_not_captured_module_bindings",
    "parameter_sizes_do_not_prove_types_struct_members_or_access_ranges",
    "opaque_library_kernels_require_public_operation_contracts",
    "source_models_require_verified_runtime_scalar_inputs",
    "terminal_coverage_device_ordering_and_live_budget_remain_unproven",
    "cpu_offload_observations_are_not_xvram_oversubscription",
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")


def _require(condition, code):
    if not condition:
        raise ValueError("binary_report_" + code)


def _object(value, fields):
    _require(type(value) is dict and set(value) == set(fields.split()), "fields")


def _array(value, maximum, minimum=0):
    _require(type(value) is list and minimum <= len(value) <= maximum, "array")


def _uint(value, maximum=MAX_U64, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum, "integer")
    return value


def _bool(value):
    _require(type(value) is bool, "boolean")


def _constant(value, expected):
    _require(type(value) is type(expected) and value == expected, "constant")


def _hash(value):
    _require(type(value) is str and _HASH.fullmatch(value) is not None, "hash")


def _sum(values):
    result = sum(values)
    _require(result <= MAX_U64, "counter_overflow")
    return result


def _layout(candidate, module):
    _object(candidate, "module_index cubin_sha256 symbol_index section_index code_bytes parameter_bytes parameters")
    _hash(candidate["cubin_sha256"])
    _require(candidate["cubin_sha256"] == module["sha256"], "module_reference")
    _uint(candidate["symbol_index"], 99999)
    _uint(candidate["section_index"], 65534, 1)
    _uint(candidate["code_bytes"], module["size_bytes"])
    bank = _uint(candidate["parameter_bytes"], 65536, 1)
    parameters = candidate["parameters"]
    _array(parameters, 128, 1)
    end = 0
    # Leading and interior ABI padding are allowed. Ordinals, nonoverlap and
    # the terminal bank extent must all match the tool's reported layout.
    for ordinal, parameter in enumerate(parameters):
        _object(parameter, "ordinal offset_bytes size_bytes")
        _uint(parameter["ordinal"], 127)
        offset = _uint(parameter["offset_bytes"], 65535)
        size = _uint(parameter["size_bytes"], 65536, 1)
        _require(parameter["ordinal"] == ordinal, "parameter_ordinals")
        _require(end <= offset and size <= bank - offset, "parameter_ranges")
        end = offset + size
    _require(end == bank, "parameter_extent")


def validate_report(report, *, allow_pending_cleanup=False):
    """Raise ValueError with a fixed code on structural/privacy/semantic failure.

    A completed worker may be validated before its owning controller has reaped
    it only with ``allow_pending_cleanup=True``. This accepts an entirely false
    cleanup ledger, never a partially completed ledger. Final completed reports
    require all cleanup flags true. Failure reports may honestly retain partial
    cleanup and never claim execution readiness or complete profile coverage.
    The caller still owns input-byte limits and process containment.
    """
    _bool(allow_pending_cleanup)
    _object(report, "schema_version report_type architecture version provenance modules kernels coverage tools decision outcome cleanup diagnostics limitations")
    _constant(report["schema_version"], SCHEMA_VERSION)
    _constant(report["report_type"], REPORT_TYPE)
    _constant(report["architecture"], "sm_86")
    _constant(report["version"], "0.1.0-dev")
    _array(report["limitations"], len(LIMITATIONS), len(LIMITATIONS))
    _require(report["limitations"] == list(LIMITATIONS), "limitations")
    diagnostics = report["diagnostics"]
    _array(diagnostics, 64)
    for code in diagnostics:
        _require(type(code) is str and len(code) <= 128 and _CODE.fullmatch(code)
                 and not _HEX_ADDRESS.search(code), "diagnostic_privacy")

    decision = report["decision"]
    _object(decision, "verdict execution_ready oversubscription_proof")
    _constant(decision["verdict"], "NO-GO")
    _constant(decision["execution_ready"], False)
    _constant(decision["oversubscription_proof"], False)
    outcome = report["outcome"]
    _object(outcome, "status exit_code")
    _require(type(outcome["status"]) is str
             and outcome["status"] in {"completed", "rejected", "timeout", "failed"}, "outcome")
    _constant(outcome["exit_code"], {"completed": 0, "rejected": 23, "timeout": 26, "failed": 27}[outcome["status"]])
    completed = outcome["status"] == "completed"
    cleanup = report["cleanup"]
    _object(cleanup, "worker_reaped process_tree_drained scratch_removed")
    for value in cleanup.values():
        _bool(value)
    _require(not cleanup["process_tree_drained"] or cleanup["worker_reaped"], "cleanup_order")
    if completed:
        _require(all(cleanup.values()) or (allow_pending_cleanup and not any(cleanup.values())), "cleanup_incomplete")
        _require(not diagnostics, "success_diagnostics")

    provenance = report["provenance"]
    _object(provenance, "upstream_commit backend_name backend_sha256 tool_name tool_sha256 tool_version reports")
    _constant(provenance["upstream_commit"], COMMIT)
    _constant(provenance["backend_name"], "ggml-cuda.dll")
    _constant(provenance["tool_name"], "cuobjdump")
    sources = provenance["reports"]
    _array(sources, 16)
    pinned = provenance["backend_sha256"] is not None
    _constant(provenance["backend_sha256"], BACKEND_SHA256 if pinned else None)
    _constant(provenance["tool_sha256"], TOOL_SHA256 if pinned else None)
    _constant(provenance["tool_version"], TOOL_VERSION if pinned else None)
    _require(bool(sources) == pinned, "provenance")
    source_hashes = set()
    for source in sources:
        _object(source, "sha256 kernel_activities")
        _hash(source["sha256"])
        _uint(source["kernel_activities"])
        _require(source["sha256"] not in source_hashes, "duplicate_source")
        source_hashes.add(source["sha256"])
    source_activities = _sum(source["kernel_activities"] for source in sources)

    modules = report["modules"]
    _array(modules, 1024)
    module_by_id = {}
    for module in modules:
        _object(module, "module_index sha256 size_bytes elf_flags symbol_functions selected_functions parameter_dump_sha256")
        index = _uint(module["module_index"], 999999, 1)
        _require(index not in module_by_id, "duplicate_module")
        module_by_id[index] = module
        _hash(module["sha256"])
        _uint(module["size_bytes"], 64 * 1024 * 1024, 64)
        _uint(module["elf_flags"], (1 << 32) - 1)
        _uint(module["symbol_functions"], 100000)
        selected = _uint(module["selected_functions"], min(4096, module["symbol_functions"]))
        if selected:
            _hash(module["parameter_dump_sha256"])
        else:
            _constant(module["parameter_dump_sha256"], None)
    _require(_sum(module["size_bytes"] for module in modules) <= 512 * 1024 * 1024, "module_size_limit")

    kernels = report["kernels"]
    _array(kernels, 4096)
    names, symbols, sections = set(), set(), set()
    selections, statuses = Counter(), Counter()
    for kernel in kernels:
        _object(kernel, "name activities status candidates runtime_binding_proven memory_bounds_proven")
        name = kernel["name"]
        _require(type(name) is str and _SAFE_NAME.fullmatch(name) and not _HEX_ADDRESS.search(name), "name_privacy")
        _require(name not in names, "duplicate_kernel")
        names.add(name)
        _uint(kernel["activities"])
        _constant(kernel["runtime_binding_proven"], False)
        _constant(kernel["memory_bounds_proven"], False)
        candidates = kernel["candidates"]
        _array(candidates, 1024)
        status = "missing" if not candidates else "static_layout_only" if len(candidates) == 1 else "ambiguous"
        _constant(kernel["status"], status)
        statuses[status] += 1
        candidate_modules = set()
        for candidate in candidates:
            _object(candidate, "module_index cubin_sha256 symbol_index section_index code_bytes parameter_bytes parameters")
            index = _uint(candidate["module_index"], 999999, 1)
            _require(index in module_by_id, "module_reference")
            _require(index not in candidate_modules, "duplicate_candidate")
            candidate_modules.add(index)
            _layout(candidate, module_by_id[index])
            symbol = (index, candidate["symbol_index"])
            section = (index, candidate["section_index"])
            _require(symbol not in symbols and section not in sections, "duplicate_function_identity")
            symbols.add(symbol)
            sections.add(section)
            selections[index] += 1
    _require(_sum(kernel["activities"] for kernel in kernels) == source_activities, "activity_reconciliation")
    for index, module in module_by_id.items():
        _require(module["selected_functions"] == selections[index], "selection_reconciliation")
    _require(not (modules or kernels) or pinned, "provenance")

    coverage = report["coverage"]
    _object(coverage, "sm86_cubins candidate_modules observed_types static_layout_types missing_types ambiguous_types runtime_bound_types complete_profile_coverage")
    counts = {"sm86_cubins": len(modules), "candidate_modules": sum(bool(value) for value in selections.values()),
              "observed_types": len(kernels), "static_layout_types": statuses["static_layout_only"],
              "missing_types": statuses["missing"], "ambiguous_types": statuses["ambiguous"], "runtime_bound_types": 0}
    for key, expected in counts.items():
        _uint(coverage[key])
        _require(coverage[key] == expected, "coverage_reconciliation")
    _constant(coverage["complete_profile_coverage"], False)

    tools = report["tools"]
    _object(tools, "invocations stdout_bytes all_direct_children_reaped all_pipes_drained")
    _uint(tools["invocations"], 1027)
    _uint(tools["stdout_bytes"], 8192 + 2 * 1024 * 1024 + 1024 * 128 * 1024 * 1024)
    _bool(tools["all_direct_children_reaped"])
    _bool(tools["all_pipes_drained"])
    if not tools["invocations"]:
        _require(not tools["stdout_bytes"] and not tools["all_direct_children_reaped"]
                 and not tools["all_pipes_drained"], "tool_reconciliation")
    if completed:
        _require(pinned and modules and kernels and source_activities > 0, "success_evidence")
        _require(tools["invocations"] == 3 + counts["candidate_modules"]
                 and tools["stdout_bytes"] > 0
                 and tools["stdout_bytes"] <= 8192 + 2 * 1024 * 1024 + counts["candidate_modules"] * 128 * 1024 * 1024
                 and tools["all_direct_children_reaped"] and tools["all_pipes_drained"], "tool_reconciliation")

    size = 0
    for part in json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":")).iterencode(report):
        size += len(part)
        _require(size <= MAX_EVIDENCE_BYTES, "size_limit")
