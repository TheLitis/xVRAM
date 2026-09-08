"""Reconcile live ABI evidence with static candidates; never infer execution GO."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from .compat_audit_binary_contract import validate_report as validate_binary
from .compat_audit_binary_contract import _object, _uint, _hash as check_hash, _constant, _array, _require
from .compat_launch_probe import analyze_trace, validate_report as validate_probe
from .compat_launch_census import _unique, _name, make_report as make_census

CAP = 4 * 1024 * 1024
GAPS = {
    "memory_bounds": ["typed_invocation_scalars_not_captured", "tensor_subranges_and_aliases_not_bound",
                      "indirect_accesses_and_native_scratch_not_bounded"],
    "cubin_binding": ["native_module_to_cubin_identity_edge_missing"],
    "device_ordering": ["generation_scoped_stream_event_edges_missing",
                        "host_api_return_is_not_gpu_retirement"],
    "trace_completeness": ["producer_quiescence_barrier_missing", "terminal_gpu_drain_and_flush_unproven"],
}
STATUS = ("matching_candidate", "ambiguous_candidates", "abi_mismatch", "no_static_candidate")


def _load(path):
    with path.open("rb") as stream:
        data = stream.read(CAP + 1)
    if len(data) > CAP:
        raise ValueError("execution_input_limit")
    result = json.loads(data, object_pairs_hook=_unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("execution_nonfinite")))
    if type(result) is not dict:
        raise ValueError("execution_input_object")
    return result, hashlib.sha256(data).hexdigest()


def _hash(path, cap):
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            total += len(data)
            if total > cap:
                raise ValueError("execution_input_limit")
            digest.update(data)
    return digest.hexdigest()


def _layout(parameters):
    return [(p["offset_bytes"], p["size_bytes"]) for p in parameters]


def reconcile_layouts(layouts, kernels):
    """Name + identical ABI is only a candidate, even if exactly one matches."""
    static = {item["name"]: item["candidates"] for item in kernels}
    if len(static) != len(kernels):
        raise ValueError("execution_duplicate_static_name")
    rows = []
    for index, live in enumerate(layouts, 1):
        name = _name(live["kernel_name"])
        layout = _layout(live["parameters"])
        candidates = [{"module_index": c["module_index"], "cubin_sha256": c["cubin_sha256"],
                       "symbol_index": c["symbol_index"], "abi_match": _layout(c["parameters"]) == layout}
                      for c in static.get(name, [])]
        matches = sum(c["abi_match"] for c in candidates)
        status = ("no_static_candidate" if not candidates else "abi_mismatch" if not matches
                  else "matching_candidate" if matches == 1 else "ambiguous_candidates")
        encoded = json.dumps(layout, separators=(",", ":")).encode("ascii")
        rows.append({"live_layout_id": index, "name": name, "calls": live["calls"],
                     "parameter_count": len(layout), "layout_sha256": hashlib.sha256(encoded).hexdigest(),
                     "status": status, "candidates": candidates,
                     "binding_proven": False, "memory_bounds_proven": False})
    return rows


def validate_report(report):
    _object(report, "schema_version report_type version provenance configuration coverage kernels gates decision outcome")
    _constant(report["schema_version"], 1)
    _constant(report["report_type"], "xvram.cuda_execution_evidence")
    _constant(report["version"], "0.1.0-dev")
    _object(report["provenance"], "probe_report probe_trace census_report census_trace binary_evidence")
    for value in report["provenance"].values():
        check_hash(value)
    config = report["configuration"]
    _object(config, "model microbatch gpu_layers context_size generated_token_limit")
    for key, value in dict(model="14b", gpu_layers=8, context_size=2048, generated_token_limit=32).items():
        _constant(config[key], value)
    _uint(config["microbatch"])
    _require(config["microbatch"] in (1, 128), "microbatch")
    counts = report["coverage"]
    _object(counts, "observed_launches metadata_calls live_layouts observed_launches_reconciled " + " ".join(STATUS))
    for key, value in counts.items():
        if key == "observed_launches_reconciled":
            _require(type(value) is bool, "routing_boolean")
        else:
            _uint(value, 100000)
    _array(report["kernels"], 4096, 1)
    total = Counter()
    seen = set()
    for index, row in enumerate(report["kernels"], 1):
        _object(row, "live_layout_id name calls parameter_count layout_sha256 status candidates binding_proven memory_bounds_proven")
        _constant(row["live_layout_id"], index)
        _name(row["name"]); check_hash(row["layout_sha256"])
        key = row["name"], row["layout_sha256"]
        _require(key not in seen, "duplicate_live_layout")
        seen.add(key)
        _uint(row["calls"], 100000, 1); _uint(row["parameter_count"], 256)
        _constant(row["binding_proven"], False); _constant(row["memory_bounds_proven"], False)
        _array(row["candidates"], 1024)
        modules, matches = set(), 0
        for candidate in row["candidates"]:
            _object(candidate, "module_index cubin_sha256 symbol_index abi_match")
            _uint(candidate["module_index"], 999999, 1)
            _uint(candidate["symbol_index"], 99999)
            check_hash(candidate["cubin_sha256"])
            _require(candidate["module_index"] not in modules and type(candidate["abi_match"]) is bool, "candidate")
            modules.add(candidate["module_index"])
            matches += candidate["abi_match"]
        expected = ("no_static_candidate" if not modules else "abi_mismatch" if not matches
                    else "matching_candidate" if matches == 1 else "ambiguous_candidates")
        _constant(row["status"], expected)
        total[expected] += 1
        total["metadata_calls"] += row["calls"]
    _require(counts["live_layouts"] == len(report["kernels"]), "layout_count")
    for key in (*STATUS, "metadata_calls"):
        _require(counts[key] == total[key], "execution_reconciliation")
    _require(counts["observed_launches"] > 0 and (not counts["observed_launches_reconciled"]
             or counts["metadata_calls"] == counts["observed_launches"]), "routing_count")
    _object(report["gates"], " ".join(GAPS))
    for key, mandatory in GAPS.items():
        gate = report["gates"][key]
        _object(gate, "proven blockers"); _constant(gate["proven"], False)
        expected = list(mandatory)
        if key == "trace_completeness" and not counts["observed_launches_reconciled"]:
            expected.append("observed_launches_not_fully_reconciled")
        if key == "cubin_binding":
            for status, code in (("no_static_candidate", "static_candidates_missing"),
                                 ("abi_mismatch", "live_static_abi_mismatch"),
                                 ("ambiguous_candidates", "multiple_static_abi_candidates")):
                if counts[status]:
                    expected.append(code)
        _require(gate["blockers"] == expected, "execution_blockers")
    _object(report["decision"], "verdict execution_ready oversubscription_proof")
    for key, value in dict(verdict="NO-GO", execution_ready=False, oversubscription_proof=False).items():
        _constant(report["decision"][key], value)
    _object(report["outcome"], "status exit_code")
    _constant(report["outcome"]["status"], "completed"); _constant(report["outcome"]["exit_code"], 0)


def analyze(probe_report, probe_trace, census_report, census_trace, binary_evidence):
    paths = dict(probe_report=probe_report, probe_trace=probe_trace, census_report=census_report,
                 census_trace=census_trace, binary_evidence=binary_evidence)
    probe, ph = _load(probe_report)
    census, ch = _load(census_report)
    binary, bh = _load(binary_evidence)
    validate_probe(probe)
    validate_binary(binary)
    if probe["mode"] != "metadata" or probe["exit_code"] != 0 or binary["outcome"]["status"] != "completed":
        raise ValueError("execution_completed_metadata_required")
    if probe["binary_sha256"].get("ggml-cuda.dll") != binary["provenance"]["backend_sha256"]:
        raise ValueError("execution_backend_provenance")
    live = analyze_trace(probe_trace)
    if live != probe["trace"] or make_census(probe, census_trace) != census or census["exit_code"] != 0:
        raise ValueError("execution_trace_report_mismatch")
    rows = reconcile_layouts(live["layouts"], binary["kernels"])
    counts = census["observation"]["counts"]
    totals = Counter(row["status"] for row in rows)
    routing = (counts["gpu_kernels_observed"] == counts["marked_kernels"]
               == counts["probe_calls_observed"] == counts["probe_calls_with_gpu_activity"]
               and not counts["open_api_calls"] and not counts["buffers_outstanding"])
    gaps = {key: list(value) for key, value in GAPS.items()}
    if not routing:
        gaps["trace_completeness"].append("observed_launches_not_fully_reconciled")
    if totals["no_static_candidate"]:
        gaps["cubin_binding"].append("static_candidates_missing")
    if totals["abi_mismatch"]:
        gaps["cubin_binding"].append("live_static_abi_mismatch")
    if totals["ambiguous_candidates"]:
        gaps["cubin_binding"].append("multiple_static_abi_candidates")
    provenance = dict(probe_report=ph, probe_trace=live["sha256"], census_report=ch,
                      census_trace=census["observation"]["sha256"], binary_evidence=bh)
    # Reject evidence modified during analysis; static evidence is deliberately a
    # different capture, never joined to a live native module by name or time.
    for key, path in paths.items():
        cap = 256 * 1024 * 1024 if key == "census_trace" else 128 * 1024 * 1024 if key == "probe_trace" else CAP
        if _hash(path, cap) != provenance[key]:
            raise ValueError("execution_input_changed")
    report = {"schema_version": 1, "report_type": "xvram.cuda_execution_evidence", "version": "0.1.0-dev",
            "provenance": provenance, "configuration": probe["configuration"],
            "coverage": {"observed_launches": counts["gpu_kernels_observed"], "metadata_calls": live["counts"]["metadata_calls"],
                         "live_layouts": len(rows), **{s: totals[s] for s in STATUS},
                         "observed_launches_reconciled": routing},
            "kernels": rows, "gates": {key: {"proven": False, "blockers": value} for key, value in gaps.items()},
            "decision": {"verdict": "NO-GO", "execution_ready": False, "oversubscription_proof": False},
            "outcome": {"status": "completed", "exit_code": 0}}
    validate_report(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("probe-report", "probe-trace", "census-report", "census-trace", "binary-evidence"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.probe_report, args.probe_trace, args.census_report, args.census_trace, args.binary_evidence)
        data = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if len(data.encode("utf-8")) > CAP:
            raise ValueError("execution_output_limit")
        with args.json.open("x", encoding="utf-8") as stream:
            stream.write(data)
        print("Execution evidence reconciled; four proof gates remain unproven.")
        return 0
    except (ValueError, KeyError, TypeError, UnicodeError, OSError, RecursionError) as error:
        print("execution evidence failure: " + type(error).__name__, file=sys.stderr)
        return 74 if isinstance(error, OSError) else 23


if __name__ == "__main__":
    raise SystemExit(main())
