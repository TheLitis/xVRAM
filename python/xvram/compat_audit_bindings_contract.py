"""Dependency-free, fail-closed launch identity evidence contract."""
import json
from .compat_audit_binary_contract import (
    _object, _array, _uint, _bool, _constant, _hash, _require, _CODE, _HEX_ADDRESS,
)

REPORT_TYPE = "xvram.cuda_compat_launch_bindings"
COUNTERS = ("gpu_activities", "correlated_gpu_activities", "loaded_modules", "hashed_modules",
            "static_module_matches", "function_activities", "native_function_lookups",
            "native_launch_links", "proven_launch_bindings")
LIMITATIONS = [
    "native_and_cupti_module_namespaces_have_no_documented_bridge_in_this_observer",
    "kernel_activity_has_no_module_or_function_id",
    "name_time_nesting_or_unique_candidate_are_not_binding_proof",
    "function_index_is_not_assumed_to_be_elf_symbol_index",
    "incomplete_finalization_cannot_prove_complete_profile_coverage",
    "no_argument_memory_ordering_or_oversubscription_proof",
]

def empty_report():
    return {"schema_version": 1, "report_type": REPORT_TYPE, "version": "0.1.0-dev",
            "provenance": {"trace_sha256": None, "binary_evidence_sha256": None},
            "coverage": dict.fromkeys(COUNTERS, 0), "modules": [], "native_links": [],
            "integrity": {"input_valid": False, "launch_records_reconciled": False,
                          "trace_complete": False, "issues": []},
            "decision": {"verdict": "NO-GO", "execution_ready": False,
                         "complete_binding_coverage": False, "oversubscription_proof": False},
            "outcome": {"status": "failed", "exit_code": 27},
            "cleanup": {"worker_reaped": False, "process_tree_drained": False, "scratch_removed": False},
            "diagnostics": [], "limitations": list(LIMITATIONS)}

def validate_report(report, *, allow_pending_cleanup=False):
    _object(report, "schema_version report_type version provenance coverage modules native_links integrity decision outcome cleanup diagnostics limitations")
    _constant(report["schema_version"], 1); _constant(report["report_type"], REPORT_TYPE)
    _constant(report["version"], "0.1.0-dev")
    _require(report["limitations"] == LIMITATIONS, "binding_limitations")
    _object(report["provenance"], "trace_sha256 binary_evidence_sha256")
    for value in report["provenance"].values():
        if value is not None: _hash(value)
    coverage = report["coverage"]
    _object(coverage, " ".join(COUNTERS))
    for value in coverage.values(): _uint(value)
    _constant(coverage["proven_launch_bindings"], 0)
    _require(coverage["correlated_gpu_activities"] <= coverage["gpu_activities"], "binding_correlation_count")
    _object(report["decision"], "verdict execution_ready complete_binding_coverage oversubscription_proof")
    _constant(report["decision"]["verdict"], "NO-GO")
    for key in ("execution_ready", "complete_binding_coverage", "oversubscription_proof"):
        _constant(report["decision"][key], False)
    seen = set()
    _array(report["modules"], 65536)
    for module in report["modules"]:
        _object(module, "context_id module_id module_generation source_sequence unload_sequence size_bytes sha256 static_module_indices")
        for key in ("context_id", "module_id", "module_generation", "source_sequence", "size_bytes"):
            _uint(module[key])
        _require(module["module_id"] > 0 and module["module_generation"] > 0 and module["source_sequence"] > 0, "binding_module_id")
        identity = (module["context_id"], module["module_id"], module["module_generation"])
        _require(identity not in seen, "binding_duplicate_module"); seen.add(identity)
        if module["unload_sequence"] is not None:
            _uint(module["unload_sequence"], minimum=module["source_sequence"]+1)
        if module["sha256"] is not None: _hash(module["sha256"])
        indices = module["static_module_indices"]
        _array(indices, 1024)
        for index in indices: _uint(index)
        _require(indices == sorted(set(indices)) and (not indices or module["sha256"] is not None), "binding_static_indices")
    _require(coverage["loaded_modules"] == len(seen), "binding_module_count")
    _require(coverage["hashed_modules"] == sum(m["sha256"] is not None for m in report["modules"]), "binding_hash_count")
    _require(coverage["static_module_matches"] == sum(bool(m["static_module_indices"]) for m in report["modules"]), "binding_static_count")
    _array(report["native_links"], 10000)
    identities = set()
    for link in report["native_links"]:
        _object(link, "context_id function_id function_generation module_id module_generation lookup_sequence first_launch_sequence last_launch_sequence launches")
        for key, value in link.items(): _uint(value, minimum=0 if key=="context_id" else 1)
        identity = tuple(link[key] for key in ("context_id", "function_id", "function_generation"))
        _require(identity not in identities, "binding_duplicate_native_link"); identities.add(identity)
        _require(link["lookup_sequence"] < link["first_launch_sequence"] <= link["last_launch_sequence"], "binding_native_interval")
    _require(coverage["native_launch_links"] == sum(link["launches"] for link in report["native_links"]), "binding_native_count")
    integrity = report["integrity"]
    _object(integrity, "input_valid launch_records_reconciled trace_complete issues")
    for key in ("input_valid", "launch_records_reconciled", "trace_complete"): _bool(integrity[key])
    _array(integrity["issues"], 256)
    _array(report["diagnostics"], 64)
    for code in integrity["issues"] + report["diagnostics"]:
        _require(type(code) is str and len(code)<=128 and _CODE.fullmatch(code) and not _HEX_ADDRESS.search(code), "binding_code")
    _require(not integrity["trace_complete"] or (integrity["input_valid"] and not integrity["issues"]), "binding_completeness")
    _object(report["outcome"], "status exit_code")
    status=report["outcome"]["status"]
    _require(type(status) is str and status in ("completed", "rejected", "timeout", "failed"), "binding_status")
    _constant(report["outcome"]["exit_code"], {"completed":0,"rejected":23,"timeout":26,"failed":27}[status])
    _object(report["cleanup"], "worker_reaped process_tree_drained scratch_removed")
    cleanup=report["cleanup"]
    for value in cleanup.values(): _bool(value)
    _require(not cleanup["process_tree_drained"] or cleanup["worker_reaped"], "binding_cleanup_order")
    _require(not cleanup["scratch_removed"] or cleanup["process_tree_drained"], "binding_scratch_order")
    if status=="completed":
        _require(all(cleanup.values()) or (allow_pending_cleanup and not any(cleanup.values())), "binding_cleanup")
        _require(not report["diagnostics"] and all(report["provenance"].values()), "binding_completion")
    _require(len(json.dumps(report, ensure_ascii=True).encode()) <= 4*1024*1024, "binding_report_limit")
