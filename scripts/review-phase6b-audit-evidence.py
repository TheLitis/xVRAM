"""Reconcile captured Phase 6b.0 evidence without rerunning CUDA or changing originals.

Outputs are new, derived observations under the matrix's fresh review/ directory.
Native CPU-offload success and profiler process success are distinct from GO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    if path.is_symlink() or path.stat().st_size > 8 * 1024**2:
        raise ValueError("invalid_or_oversized_json_artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    import jsonschema
    from xvram.compat_audit_analysis import analyze, combine_gguf_metadata, read_gguf_metadata
    from xvram.compat_audit_nsys import summarize_nsys

    matrix_path = args.matrix.resolve()
    directory = matrix_path.parent
    output = directory / "review"
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise ValueError("review_directory_must_be_fresh_and_empty")
    matrix = read_json(matrix_path)
    profile_path = root / "python/xvram/compat_audit_profile.json"
    profile = read_json(profile_path)
    expected_cases = {f"{model}-ub{batch}-{mode}" for model in ("14b", "32b")
                      for batch in (1, 128) for mode in ("baseline", "cupti")}
    expected_cases.update(f"{model}-ub128-nsys" for model in ("14b", "32b"))
    cases = matrix["cases"]
    if len(cases) != 10 or {case["case"] for case in cases} != expected_cases:
        raise ValueError("full_ten_case_matrix_required")
    verified: dict[str, str] = {"matrix.json": digest(matrix_path)}
    captures, reports = {}, {}
    # Validate every originally published artifact hash before deriving anything.
    for case in cases:
        name = case["case"]
        artifact_hashes = dict(case["artifacts"])
        for filename, wanted in (("report.json", case["report_sha256"]),
                                 ("capture.json", case["capture_sha256"])):
            if artifact_hashes.get(filename) != wanted:
                raise ValueError("contradictory_matrix_artifact_hash")
        for filename, wanted in artifact_hashes.items():
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("nonlocal_matrix_artifact_name")
            path = directory / name / filename
            if path.is_symlink() or digest(path) != wanted:
                raise ValueError("original_artifact_hash_mismatch")
            verified[f"{name}/{filename}"] = wanted
        captures[name] = read_json(directory / name / "capture.json")
        reports[name] = read_json(directory / name / "report.json")
        capture = captures[name]
        if (capture["profile_id"] != profile["profile_id"] or capture["model"] != case["model"]
                or capture["capture_mode"] != case["mode"]
                or capture["configuration"]["microbatch"] != case["microbatch"]):
            raise ValueError("matrix_capture_configuration_mismatch")

    model_metadata = {}
    for model in ("14b", "32b"):
        shard_paths = [args.dependencies / f"qwen{model}" / item["name"]
                       for item in profile["models"][model]["files"]]
        model_metadata[model] = combine_gguf_metadata([read_gguf_metadata(path) for path in shard_paths])
    source_files = [Path(__file__).resolve(), profile_path,
                    root / "python/xvram/compat_audit_analysis.py",
                    root / "python/xvram/compat_audit_nsys.py",
                    root / "schemas/cuda-compat-audit-v1.schema.json"]
    source_hashes = {str(path.relative_to(root)).replace("\\", "/"): digest(path) for path in source_files}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=True, timeout=10).stdout.strip()
    validator = jsonschema.Draft202012Validator(read_json(root / "schemas/cuda-compat-audit-v1.schema.json"))
    output.mkdir(exist_ok=True)
    rows, derived = [], {}
    for case in cases:
        name, model, mode = case["case"], case["model"], case["mode"]
        capture, original = captures[name], reports[name]
        run = capture["run"]
        command_ok = (case["exit_code"] == 0 and run["exit_code"] == 0
                      and run["controller_reaped"] is True and run["process_tree_drained"] is True
                      and not run["errors"] and not run["output_truncated"] and not run["timed_out"])
        row = {"case": name, "model": model, "microbatch": case["microbatch"], "mode": mode,
               "native_exit_confirmed": command_ok if mode in ("baseline", "cupti") else False,
               "profiler_command_completed": command_ok if mode == "nsys" else None,
               "native_cpu_offload": True, "oversubscription_proof": False,
               "original_report_sha256": case["report_sha256"],
               "original_capture_sha256": case["capture_sha256"],
               "process_tree_drained": run["process_tree_drained"],
               "stdout_sha256": run["stdout_sha256"] if mode != "nsys" else None,
               "baseline_measurements": {key: run.get(key) for key in
                   ("elapsed_ms", "ttft_ms", "ttft_status", "timings", "memory", "gpu")} if mode == "baseline" else None,
               "full_trace": False, "kernel_memory_contracts": "unresolved"}
        if mode == "cupti":
            metadata = model_metadata[model]
            if any(original["model"][key] != metadata[key] for key in
                   ("metadata_sha256", "file_size_bytes", "tensor_count")):
                raise ValueError("captured_model_metadata_mismatch")
            inventory = original["inventory"]
            if {item["name"]: item["sha256"] for item in inventory} != profile["upstream"]["files"]:
                raise ValueError("captured_binary_inventory_mismatch")
            provenance = dict(original["provenance"])
            provenance["tool_versions"] = {"review_source_revision": revision,
                "original_report_sha256": case["report_sha256"],
                "analyzer_sha256": source_hashes["python/xvram/compat_audit_analysis.py"],
                "review_script_sha256": source_hashes["scripts/review-phase6b-audit-evidence.py"]}
            derived_report = analyze([directory / name / "cupti-trace.jsonl"], inventory, metadata,
                {"profile_id": profile["profile_id"], "cache_budget_bytes": None,
                 "scratch_reserve_bytes": None, "source_evidence": profile["source_evidence"]},
                provenance=provenance, cleanup={"controller_reaped": run["controller_reaped"],
                                               "collector_finalized": None, "complete": None})
            validator.validate(derived_report)
            if derived_report["decision"]["verdict"] != "NO-GO":
                raise ValueError("unexpected_readiness_requires_manual_review")
            filename = name + "-report.json"
            write_json(output / filename, derived_report)
            derived[filename] = digest(output / filename)
            coverage = derived_report["coverage"]
            row.update(derived_report_file=filename, derived_report_sha256=derived[filename],
                       decision=derived_report["decision"], coverage=coverage,
                       memory_bounds=derived_report["memory_bounds"], unresolved=derived_report["unresolved"])
        elif mode == "nsys":
            sqlite_path = directory / name / "native-nsight.sqlite"
            summary = summarize_nsys(sqlite_path)
            original_summary = read_json(directory / name / "nsight-summary.json")
            if summary["source"] != original_summary["source"]:
                raise ValueError("nsight_source_hash_mismatch")
            filename = name + "-summary.json"
            write_json(output / filename, summary)
            derived[filename] = digest(output / filename)
            row.update(derived_summary_file=filename, derived_summary_sha256=derived[filename],
                       counts=summary["counts"], api_class_counts=summary["api_class_counts"],
                       profiler_diagnostics=summary["profiler_diagnostics"], unresolved=summary["unresolved"],
                       native_exit_scope="direct_command_was_profiler_not_target_application")
        rows.append(row)
        print(f"Reviewed {name}: original hashes valid, full_trace=False", flush=True)
    agreements = {}
    for model in ("14b", "32b"):
        selected = [row for row in rows if row["model"] == model and row["mode"] != "nsys"]
        hashes = {row["stdout_sha256"] for row in selected}
        agreements[model] = {"direct_cases": len(selected), "equal": len(hashes) == 1,
                             "sha256": next(iter(hashes)) if len(hashes) == 1 else None,
                             "scope": "sanitized_visible_output_not_tensor_or_numerical_equivalence"}
    # Recheck input bytes after analysis. Original files are never written.
    for relative, expected in verified.items():
        if digest(directory / relative) != expected:
            raise ValueError("original_artifact_changed_during_review")
    review = {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_review",
              "profile_id": profile["profile_id"], "original_matrix_sha256": verified["matrix.json"],
              "original_hashes_valid": True, "original_artifact_hashes": verified,
              "derived_artifact_hashes": derived, "review_source_revision": revision,
              "review_source_hashes": source_hashes, "inventory_scope": "captured_hash_pinned_inventory_not_rescanned",
              "model_recheck_scope": "GGUF_metadata_only_weight_payload_hashes_not_rechecked",
              "full_trace": False, "decision": "NO-GO", "interoperability_impossible_proven": False,
              "oversubscription_proof": False, "native_cpu_offload": True,
              "nsys_original_matrix_clarification": "native_observation_completed_meant_profiler_command_success_not_confirmed_native_target_exit",
              "visible_output_digest_agreement": agreements, "cases": rows}
    write_json(output / "review.json", review)
    print(json.dumps({"reports": 4, "nsight_summaries": 2, "original_hashes_valid": True,
                      "digest_agreement": agreements, "decision": "NO-GO"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
