"""Offline shape, semantic and exact-input validation for a same-process census."""
import argparse
import json
from pathlib import Path

import jsonschema
from xvram.compat_launch_census import make_report, validate_report
from xvram.compat_launch_probe import analyze_trace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--probe-report", type=Path, required=True)
parser.add_argument("--probe-trace", type=Path, required=True)
parser.add_argument("--census-trace", type=Path, required=True)
args = parser.parse_args()
schemas = Path(__file__).resolve().parents[1] / "schemas"
for path in (args.report, args.probe_report):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("report_limit")
report = json.loads(args.report.read_text())
probe = json.loads(args.probe_report.read_text())
jsonschema.validate(probe, json.loads((schemas / "cuda-launch-probe-report-v1.schema.json").read_text()))
jsonschema.validate(report, json.loads((schemas / "cuda-launch-census-report-v1.schema.json").read_text()))
validate_report(report)
if analyze_trace(args.probe_trace) != probe["trace"] or make_report(probe, args.census_trace) != report:
    raise ValueError("census_input_mismatch")
trace_schema = json.loads((schemas / "cuda-launch-census-trace-v1.schema.json").read_text())
validators = {variant["properties"]["kind"]["const"]: jsonschema.Draft202012Validator(variant)
              for variant in trace_schema["oneOf"]}
with args.census_trace.open("rb") as stream:
    while raw := stream.readline(65537):
        if len(raw) > 65536:
            raise ValueError("trace_record_limit")
        record = json.loads(raw)
        validators[record["kind"]].validate(record)
print("census shape, semantics and exact inputs validated; terminal completeness and execution readiness remain unproven")
