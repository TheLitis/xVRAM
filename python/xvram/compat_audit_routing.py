"""Bounded observations of CUDA public entry-point resolution, never interposition.

Resolver metadata is independent of invocation routing and kernel memory bounds.
A successfully returned function ID proves neither which module owns it nor that
the application invoked it. No native library is loaded by this module.
"""
from __future__ import annotations

from collections import Counter

from . import compat_audit_analysis as audit


RESOLVERS = {
    ("driver", "cuGetProcAddress"),
    ("driver", "cuGetProcAddress_v2"),
    ("runtime", "cudaGetDriverEntryPoint"),
    ("runtime", "cudaGetDriverEntryPoint_ptsz"),
    ("runtime", "cudaGetDriverEntryPointByVersion"),
    ("runtime", "cudaGetDriverEntryPointByVersion_ptsz"),
}
MAX_ROUTING_ITEMS = 16384
MAX_ROUTING_TEXT_BYTES = 2 * 1024 * 1024
_INPUT_FIELDS = ("requested_symbol", "requested_version", "resolver_flags")
_SUMMARY_ERRORS = ("dropped_records", "unknown_callback_details", "unknown_activity_kinds",
                   "serialization_errors", "collector_errors", "incomplete_activities")


def summarize_routing(trace_path):
    """Summarize one strict trace. ``resolver_calls`` counts observed API exits.

    Unknown/malformed/incomplete observations remain unresolved. Counts describe
    callback layers separately; Runtime and nested Driver resolution requests are
    not deduplicated into an invented number of application routing operations.
    """
    issues, counts, records = Counter(), Counter(), Counter()
    pending, seen, requests = {}, set(), {}
    text_bytes = entry_ids = 0
    sequence, timestamp, version, terminal = 0, 0, None, None
    sessions = 0
    try:
        for record in audit._trace_records(trace_path):
            if sum(map(len, (pending, seen, requests))) + entry_ids > min(MAX_ROUTING_ITEMS, audit.MAX_STATE_ITEMS):
                raise audit.AuditInputError("routing_state_limit")
            if record["sequence"] != sequence + 1:
                issues["sequence_gap"] += 1
            if record["timestamp_ns"] < timestamp:
                issues["non_monotonic_timestamp"] += 1
            sequence, timestamp = record["sequence"], record["timestamp_ns"]
            kind = record["kind"]
            records[kind] += 1
            if terminal is not None:
                issues["records_after_summary"] += 1
            if kind == "session":
                sessions += 1
                if sessions != 1 or sequence != 1:
                    issues["invalid_session_order"] += 1
                version = record["schema_version"]
            elif version != record["schema_version"]:
                issues["mixed_trace_versions_or_missing_session"] += 1
            if kind == "summary":
                terminal = record
                continue
            if kind == "gap":
                issues["collector_gap"] += 1
            if kind not in ("api_enter", "api_exit"):
                continue
            if (record["domain"], record["symbol"]) not in RESOLVERS:
                continue
            key = (record["domain"], record["correlation_id"], record["thread_id"])
            inputs = tuple(record.get(field) for field in _INPUT_FIELDS)
            if kind == "api_enter":
                if key in seen:
                    issues["duplicate_resolver_identity"] += 1
                else:
                    seen.add(key)
                if key in pending:
                    issues["duplicate_resolver_enter"] += 1
                pending[key] = (record["symbol"], inputs, record["detail_known"])
                continue
            counts["resolver_calls"] += 1
            entered = pending.pop(key, None)
            if entered is None or entered[0] != record["symbol"]:
                issues["unmatched_resolver_exit"] += 1
                counts["unknown_calls"] += 1
                continue
            if record["schema_version"] not in (2, 3) or not entered[2] or not record["detail_known"] or inputs[0] is None or inputs[2] is None:
                issues["resolver_metadata_unavailable"] += 1
                counts["unknown_calls"] += 1
                continue
            if inputs != entered[1]:
                issues["resolver_input_changed"] += 1
                counts["unknown_calls"] += 1
                continue
            request_key = (record["domain"], record["symbol"], *inputs)
            if request_key not in requests:
                text_bytes += len(inputs[0].encode("utf-8"))
                if text_bytes > MAX_ROUTING_TEXT_BYTES:
                    raise audit.AuditInputError("routing_text_limit")
                requests[request_key] = {"calls": 0, "resolved_calls": 0, "entry_points": set()}
            item = requests[request_key]
            item["calls"] += 1
            if record["status"] != 0:
                counts["api_failed_calls"] += 1
                continue
            query = record.get("query_status")
            if query is not None and query != 0:
                counts["query_failed_calls"] += 1
                if query not in (1, 2):
                    issues["unknown_resolver_query_status"] += 1
                if "entry_point_id" in record:
                    issues["failed_query_has_entry_point"] += 1
                continue
            point = record.get("entry_point_id")
            if point is None:
                counts["unknown_calls"] += 1
                issues["successful_resolver_without_entry_point"] += 1
                continue
            counts["resolved_calls"] += 1
            item["resolved_calls"] += 1
            if point not in item["entry_points"]:
                item["entry_points"].add(point)
                entry_ids += 1
    except audit.AuditInputError as error:
        # The parser emits fixed codes, never a filename or native value. Do not
        # copy arbitrary exception text into the derived diagnostic artifact.
        code = str(error)
        issues[code if code in ("routing_state_limit", "routing_text_limit", "trace_limit") else "invalid_routing_trace"] += 1
    if pending:
        issues["unclosed_resolver_calls"] += len(pending)
    if sessions != 1:
        issues["missing_or_multiple_sessions"] += 1
    if terminal is None:
        issues["missing_summary"] += 1
    else:
        if not terminal["complete"] or not terminal["safely_finalized"] or terminal["terminal_checkpoint"] != "explicit_finalize":
            issues["collector_not_safely_finalized"] += 1
        if any(terminal[field] for field in _SUMMARY_ERRORS):
            issues["collector_incomplete_evidence"] += 1
        if terminal["buffers_requested"] != terminal["buffers_completed"]:
            issues["uncompleted_activity_buffers"] += 1
        expected = {"callback_records": records["api_enter"] + records["api_exit"],
                    "activity_records": records["activity"], "resource_records": records["resource"]}
        if any(terminal[field] != value for field, value in expected.items()):
            issues["summary_counter_mismatch"] += 1
    if counts["resolver_calls"] == 0:
        issues["no_resolver_observations"] += 1
    result = {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_routing",
              **{field: counts[field] for field in ("resolver_calls", "resolved_calls", "api_failed_calls",
                                                   "query_failed_calls", "unknown_calls")},
              "requests": [], "issues": [{"code": code, "count": count} for code, count in sorted(issues.items())],
              "evidence": "unresolved" if issues else "observed",
              "routing_coverage_complete": False, "semantic_ranges_proven": False}
    for key, item in sorted(requests.items(), key=lambda pair: tuple("" if value is None else str(value) for value in pair[0])):
        domain, api, symbol, requested_version, flags = key
        result["requests"].append({"domain": domain, "api": api, "requested_symbol": symbol,
                                   "requested_version": requested_version, "resolver_flags": flags,
                                   "calls": item["calls"], "resolved_calls": item["resolved_calls"],
                                   "entry_point_count": len(item["entry_points"])})
    return result
