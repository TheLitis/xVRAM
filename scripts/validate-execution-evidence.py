"""Recompute an execution-evidence report from its bounded exact inputs."""
import argparse
import json
from pathlib import Path

import jsonschema
from xvram.compat_audit_execution import _load, analyze, validate_report

parser = argparse.ArgumentParser(description=__doc__)
for key in ("probe-report", "probe-trace", "census-report", "census-trace", "binary-evidence", "report"):
    parser.add_argument("--" + key, type=Path, required=True)
args = parser.parse_args()
report, _ = _load(args.report)
schema = Path(__file__).resolve().parents[1] / "schemas/cuda-execution-evidence-v1.schema.json"
jsonschema.validate(report, json.loads(schema.read_text()))
validate_report(report)
if report != analyze(args.probe_report, args.probe_trace, args.census_report, args.census_trace, args.binary_evidence):
    raise ValueError("execution_evidence_input_mismatch")
print("Execution evidence schema, semantics and exact inputs validated; NO-GO remains.")
