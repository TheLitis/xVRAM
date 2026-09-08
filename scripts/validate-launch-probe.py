"""Validate a completed or failed launch-probe report and its optional trace, offline."""
import argparse
import json
from pathlib import Path

import jsonschema
from xvram.compat_launch_probe import analyze_trace, validate_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--trace", type=Path)
args = parser.parse_args()
schemas = Path(__file__).resolve().parents[1] / "schemas"
if args.report.stat().st_size > 4 * 1024 * 1024:
    raise ValueError("report_limit")
report = json.loads(args.report.read_text(encoding="utf-8"))
jsonschema.validate(report, json.loads((schemas / "cuda-launch-probe-report-v1.schema.json").read_text()))
validate_report(report)
if args.trace:
    if analyze_trace(args.trace) != report["trace"]:
        raise ValueError("trace_report_mismatch")
    validator = jsonschema.Draft202012Validator(json.loads((schemas / "cuda-launch-probe-trace-v1.schema.json").read_text()))
    with args.trace.open("rb") as stream:
        while line := stream.readline(65537):
            if len(line) > 65536:
                raise ValueError("record_limit")
            validator.validate(json.loads(line))
print("launch probe schema, semantics and supplied trace validated; no execution-readiness claim")
