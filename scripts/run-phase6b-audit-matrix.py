"""Run the pinned native observation matrix; never an xVRAM execution gate.

Artifacts are kept in a new directory. Large native Nsight artifacts remain local.
An audit decision of NO-GO is a valid finding, not a failed native baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--trace-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import jsonschema

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))
    from xvram.compat_audit_capture import run_process
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be a fresh empty directory")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root / "python"))
    schema = json.loads((root / "schemas/cuda-compat-audit-v1.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    cases = []
    # Unprofiled baselines always run separately from both profiling tools.
    for model in ("14b", "32b"):
        for microbatch in (1, 128):
            for mode in ("baseline", "cupti"):
                cases.append((model, microbatch, mode))
        if args.nsys:
            cases.append((model, 128, "nsys"))
    results = []
    for model, microbatch, mode in cases:
        name = f"{model}-ub{microbatch}-{mode}"
        directory = output / name
        command = [sys.executable, "-m", "xvram.compat_audit", "--stage", "all",
                   "--model", model, "--microbatch", str(microbatch), "--capture-mode", mode,
                   "--trace-version", str(args.trace_version),
                   "--binary-dir", str(args.dependencies / "llama-b10819"),
                   "--model-dir", str(args.dependencies / f"qwen{model}"),
                   "--collector", str(args.collector), "--gpu-layers", "8",
                   "--context-size", "2048", "--generate", "32", "--timeout-seconds", "900",
                   "--output-dir", str(directory), "--no-text"]
        if args.nsys:
            command.extend(["--nsys", str(args.nsys)])
        print(f"Running {name}: native CPU-offload observation only", flush=True)
        completed = run_process(command, environment=env, timeout_seconds=1200,
                                output_dir=output / "controllers" / name)
        report_path = directory / "report.json"
        capture_path = directory / "capture.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validator.validate(report)
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, jsonschema.ValidationError) as error:
            failure = {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_matrix",
                       "cases": results, "matrix_complete": False, "oversubscription_proof": False,
                       "failed_case": name, "failure": type(error).__name__, "controller": completed}
            (output / "matrix.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            return 26 if completed.get("timed_out") else 27
        run = capture["run"]
        command_ok = (completed["exit_code"] == 0 and completed.get("process_tree_drained") is True and run["exit_code"] == 0
                     and run["controller_reaped"] and run.get("process_tree_drained") is True
                     and not run.get("errors") and not run.get("output_truncated"))
        row = {"case": name, "model": model, "microbatch": microbatch, "mode": mode,
               "exit_code": completed["exit_code"], "command_completed": command_ok,
               "native_exit_confirmed": command_ok if mode != "nsys" else None,
               "decision": report["decision"], "report_sha256": digest(report_path),
               "capture_sha256": digest(capture_path), "measurements": run,
               "artifacts": {path.name: digest(path) for path in sorted(directory.iterdir())
                             if path.is_file() and path.suffix in (".json", ".jsonl", ".txt")}}
        results.append(row)
        manifest = {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_matrix",
                    "profile_id": capture["profile_id"], "native_cpu_offload": True,
                    "oversubscription_proof": False, "cases": results,
                    "selected_matrix_only": True, "nsight_selected": args.nsys is not None,
                    "full_observation_evidence_complete": False,
                    "matrix_complete": len(results) == len(cases) and all(
                        item["command_completed"] for item in results)}
        (output / "matrix.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{name}: command_completed={command_ok}, audit={report['decision']['verdict']}", flush=True)
        if not command_ok:
            return 27
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
