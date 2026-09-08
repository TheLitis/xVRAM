"""Validate v3 trace and linked offline evidence without a GPU or native loads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jsonschema

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from xvram.compat_audit_analysis import _trace_records  # noqa: E402
from xvram.compat_audit_bindings_contract import validate_report  # noqa: E402
from xvram.compat_audit_binary_contract import validate_report as validate_binary  # noqa: E402

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--bindings",type=Path,required=True)
    parser.add_argument("--binary-evidence",type=Path,required=True)
    args=parser.parse_args()
    schema=json.loads((ROOT/"schemas/cuda-compat-audit-trace-v3.schema.json").read_text())
    # Each oneOf branch has a unique kind const, so dispatch is equivalent to
    # testing the whole oneOf and avoids testing unrelated branches per record.
    validators={v["properties"]["kind"]["const"]:jsonschema.Draft202012Validator(v) for v in schema["oneOf"]}
    if len(validators)!=len(schema["oneOf"]): raise ValueError("ambiguous trace schema")
    digest=hashlib.sha256(); count=0
    for record in _trace_records(args.trace,digest=digest):
        validators[record["kind"]].validate(record); count+=1
    report_bytes=args.bindings.read_bytes()
    report=json.loads(report_bytes); validate_report(report)
    jsonschema.validate(report,json.loads((ROOT/"schemas/cuda-compat-launch-bindings-v1.schema.json").read_text()))
    binary_bytes=args.binary_evidence.read_bytes(); binary=json.loads(binary_bytes); validate_binary(binary)
    jsonschema.validate(binary,json.loads((ROOT/"schemas/cuda-compat-audit-binary-evidence-v1.schema.json").read_text()))
    if report["provenance"]!={"trace_sha256":digest.hexdigest(),"binary_evidence_sha256":hashlib.sha256(binary_bytes).hexdigest()}:
        raise ValueError("input hashes disagree")
    print(json.dumps({"trace_records":count,"trace_sha256":digest.hexdigest(),
                      "bindings_sha256":hashlib.sha256(report_bytes).hexdigest(),
                      "schemas_valid":True,"semantic_contract_valid":True,"decision":report["decision"]},indent=2))

if __name__=="__main__": main()
