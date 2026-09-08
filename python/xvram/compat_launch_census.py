"""Same-process CUPTI/Driver-boundary observations. Never a complete-coverage proof."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from .compat_launch_probe import analyze_trace, validate_report as validate_probe

CUPTI_NAME = "cupti64_2026.2.1.dll"
CUPTI_SHA256 = "9b10d2fafaff1a4dc9e447c4a1355fccb04ee024fa7e7d28c9e4c5ab53347faf"
CAP = 128 * 1024 * 1024
PROOF = ("all_launches_covered", "terminal_complete", "semantic_ranges", "cubin_binding", "oversubscription")
LAUNCH = re.compile(r"^(?:cuda|cu)(?:Launch|GraphLaunch)")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def _integer(value, maximum=2**64-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("invalid_integer")
    return value


def _name(value):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or re.search(r"0[xX][0-9a-fA-F]{6,}|(?<![\w])[0-9a-fA-F]{16}(?![\w])", value)):
        raise ValueError("invalid_or_private_name")
    return value


def analyze(path: Path, probe_calls: int):
    _integer(probe_calls)
    size = path.stat().st_size
    if not probe_calls or size > 2 * CAP:
        raise ValueError("census_limit")
    digest = hashlib.sha256()
    apis, active, children, kernels, depths, failed = {}, {}, Counter(), [], {}, set()
    summary = None
    sequence = 0
    total = 0
    version = None
    common = {"schema_version", "record_type", "sequence", "kind"}
    with path.open("rb") as stream:
        while raw := stream.readline(65537):
            total += len(raw)
            if len(raw) > 65536 or not raw.endswith(b"\n") or sequence >= (1500000 if version == 2 else 750000):
                raise ValueError("census_framing_limit")
            digest.update(raw)
            record = json.loads(raw, object_pairs_hook=_unique)
            sequence += 1
            if (not isinstance(record, dict) or record.get("record_type") != "xvram.cuda_launch_census"
                    or type(record.get("schema_version")) is not int or record["schema_version"] not in (1, 2)
                    or _integer(record.get("sequence")) != sequence or summary is not None):
                raise ValueError("census_sequence")
            if version is not None and record["schema_version"] != version:
                raise ValueError("census_version_changed")
            version = record["schema_version"]
            if max(size, total) > CAP * version:
                raise ValueError("census_version_capacity")
            kind = record.get("kind")
            if sequence == 1 and kind != "session":
                raise ValueError("census_session_missing")
            if kind == "session":
                if (sequence != 1 or set(record) != common | {"terminal_complete", "cupti_sha256"}
                        or record["terminal_complete"] is not False or record["cupti_sha256"] != CUPTI_SHA256):
                    raise ValueError("census_session")
            elif kind in ("api_enter", "api_exit"):
                fields = common | {"domain", "correlation_id", "probe_call_id", "symbol", "api_id", "parent_api_id"}
                if kind == "api_exit":
                    fields |= {"result"}
                    _integer(record.get("result"), 2**31-1)
                if set(record) != fields or record["domain"] not in ("runtime", "driver"):
                    raise ValueError("census_api_fields")
                key = _integer(record["api_id"])
                parent = _integer(record["parent_api_id"])
                correlation = _integer(record["correlation_id"], 2**32-1)
                value = (record["domain"], correlation, _integer(record["probe_call_id"], probe_calls),
                         _name(record["symbol"]), parent)
                if not correlation:
                    raise ValueError("census_zero_correlation")
                if kind == "api_enter":
                    if key != len(apis) + 1 or len(apis) >= 250000 * version or (parent and parent not in active):
                        raise ValueError("census_api_identity")
                    apis[key] = value
                    active[key] = value
                    depths[key] = depths.get(parent, 0) + 1
                    if depths[key] > 64:
                        raise ValueError("census_api_depth")
                    children[parent] += 1
                else:
                    if children[key] or active.pop(key, None) != value:
                        raise ValueError("census_exit_mismatch")
                    children[parent] -= 1
                    if record["result"]:
                        failed.add(key)
            elif kind == "kernel":
                if set(record) != common | {"correlation_id", "name", "start_ns", "end_ns"}:
                    raise ValueError("census_kernel_fields")
                correlation = _integer(record["correlation_id"], 2**32-1)
                start, end = _integer(record["start_ns"]), _integer(record["end_ns"])
                if not correlation or not start or end < start or len(kernels) >= 100000:
                    raise ValueError("census_kernel_bounds")
                kernels.append((correlation, _name(record["name"])))
            elif kind == "summary":
                if set(record) != common | {"errors", "dropped", "buffers_outstanding", "terminal_complete"}:
                    raise ValueError("census_summary_fields")
                for key in ("errors", "dropped", "buffers_outstanding"):
                    _integer(record[key])
                if record["terminal_complete"] is not False:
                    raise ValueError("census_false_terminal_claim")
                summary = record
            else:
                raise ValueError("census_kind")
    if not summary or not kernels or summary["errors"] or summary["dropped"]:
        raise ValueError("census_incomplete_or_lossy")
    # Buffered records may precede or follow their callbacks. Resolve after input
    # is checked. Native CUPTI IDs correlate events, not tensors or binary identity.
    groups, observed_calls = Counter(), set()
    counts = {"gpu_kernels_observed": len(kernels), "marked_kernels": 0,
              "unmarked_kernels": 0, "uncorrelated_kernels": 0,
              "api_pairs": len(apis) - len(active), "open_api_calls": len(active),
              "probe_calls_observed": probe_calls, "probe_calls_with_gpu_activity": 0,
              "buffers_outstanding": summary["buffers_outstanding"]}
    launches = {}
    for key, (domain, correlation, marker, symbol, parent) in apis.items():
        if key not in active and LAUNCH.match(symbol):
            launches.setdefault(correlation, set()).add(key)
    for correlation, name in kernels:
        candidates = launches.get(correlation, set())
        ancestors = set()
        for key in candidates:
            parent = apis[key][4]
            depth = 0
            while parent:
                ancestors.add(parent)
                parent = apis[parent][4]
                depth += 1
                if depth > 64:
                    raise ValueError("census_api_depth")
        leaves = candidates - ancestors
        if leaves & failed:
            raise ValueError("census_failed_launch_with_activity")
        markers = {apis[key][2] for key in leaves}
        if len(markers) > 1:
            raise ValueError("census_ambiguous_marker")
        ids = markers - {0}
        links = sorted({(apis[key][0], apis[key][2], apis[key][3]) for key in candidates})
        classification = "marked" if ids else "unmarked" if links else "uncorrelated"
        counts[classification + "_kernels"] += 1
        observed_calls.update(ids)
        origins = ";".join(sorted({f"{domain}:{symbol}" for domain, _, symbol in links})) or "unobserved"
        if len(origins) > 8250:
            raise ValueError("census_origin_limit")
        groups[(name, origins, classification)] += 1
    counts["probe_calls_with_gpu_activity"] = len(observed_calls)
    if len(groups) > 4096:
        raise ValueError("census_kernel_name_limit")
    return {"sha256": digest.hexdigest(), "records": sequence, "counts": counts,
            "kernels": [{"name": name, "origin": origin, "classification": classification, "count": count}
                        for (name, origin, classification), count in sorted(groups.items())]}


def make_report(probe, trace_path):
    validate_probe(probe)
    if probe["mechanism"] != "driver_entry_probe":
        raise ValueError("census_probe_provenance")
    report = {"schema_version": 1, "report_type": "xvram.cuda_launch_census", "version": "0.1.0-dev",
              "cupti_sha256": CUPTI_SHA256,
              "observation": None, "diagnostics": [], "proof": dict.fromkeys(PROOF, False),
              "exit_code": probe["exit_code"], "probe_trace_sha256": probe["trace"]["sha256"] if probe["trace"] else None}
    try:
        report["observation"] = analyze(trace_path, probe["trace"]["counts"]["calls_returned"])
    except (OSError, ValueError, TypeError, KeyError):
        report["diagnostics"].append("census_invalid_or_missing")
        if report["exit_code"] == 0:
            report["exit_code"] = 27
    validate_report(report)
    return report


def validate_report(report):
    if (report.get("schema_version") != 1 or type(report.get("schema_version")) is not int
            or report.get("report_type") != "xvram.cuda_launch_census"
            or report.get("version") != "0.1.0-dev" or report.get("cupti_sha256") != CUPTI_SHA256):
        raise ValueError("census_report_profile")
    if set(report.get("proof", {})) != set(PROOF) or any(value is not False for value in report["proof"].values()):
        raise ValueError("census_false_proof")
    if type(report.get("exit_code")) is not int or report["exit_code"] not in (0, 23, 26, 27, 64, 70, 74):
        raise ValueError("census_report_exit")
    observation = report.get("observation")
    if report["exit_code"] == 0 and (observation is None or report.get("diagnostics")):
        raise ValueError("census_false_success")
    if observation is None:
        return
    counts = observation["counts"]
    for value in counts.values():
        _integer(value)
    if (counts["gpu_kernels_observed"] != sum(counts[key] for key in ("marked_kernels", "unmarked_kernels", "uncorrelated_kernels"))
            or counts["probe_calls_with_gpu_activity"] > min(counts["probe_calls_observed"], counts["marked_kernels"])
            or not counts["gpu_kernels_observed"]):
        raise ValueError("census_report_reconciliation")
    totals, seen = Counter(), set()
    for row in observation["kernels"]:
        key = (row["name"], row["origin"], row["classification"])
        if key in seen or row["classification"] not in ("marked", "unmarked", "uncorrelated") or not _integer(row["count"]):
            raise ValueError("census_report_kernel")
        seen.add(key)
        totals[row["classification"]] += row["count"]
    if any(totals[key] != counts[key + "_kernels"] for key in ("marked", "unmarked", "uncorrelated")):
        raise ValueError("census_report_kernel_totals")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--probe-trace", type=Path, required=True)
    parser.add_argument("--census-trace", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.probe_report.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("report_limit")
        probe = json.loads(args.probe_report.read_text(), object_pairs_hook=_unique)
        validate_probe(probe)
        if analyze_trace(args.probe_trace) != probe["trace"]:
            raise ValueError("probe_trace_mismatch")
        report = make_report(probe, args.census_trace)
        with args.json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return report["exit_code"]
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"census validation/output failure: {type(error).__name__}", file=sys.stderr)
        return 74 if isinstance(error, OSError) else 23


if __name__ == "__main__":
    raise SystemExit(main())
