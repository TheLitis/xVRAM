"""Reconcile observed API nesting and GPU activities, not tensor semantics.

Correlation IDs identify CUPTI invocations. Host stack nesting establishes only
an observed Runtime/Driver relationship; it does not prove memory accesses or
device ordering. Buffered activities may precede or follow their API records.
"""
from collections import Counter, defaultdict
import hashlib

from .compat_audit_analysis import AuditInputError, MAX_STATE_ITEMS, _trace_records


_GEOMETRY = tuple(prefix + axis for prefix in ("grid_", "block_") for axis in "xyz")
_TERMINAL_ERRORS = ("dropped_records", "serialization_errors", "collector_errors",
                    "incomplete_activities", "unknown_activity_kinds")
MAX_NESTING_DEPTH = 256
_LAUNCH_FIELDS = ("callback_id", "context_id", "stream_id", "kernel_name", "shared_bytes") + _GEOMETRY


def correlate_trace(path):
    """Return bounded, address-free evidence for exactly one normalized trace."""
    issues, integrity, counts = Counter(), Counter(), Counter()
    stacks, active, launches, activities = {}, set(), {}, defaultdict(list)
    api_activities = Counter()
    sequence, timestamp, version, session, summary = 0, 0, None, None, None
    stored, input_valid = 0, True
    first_digest, second_digest = hashlib.sha256(), hashlib.sha256()
    try:
        for record in _trace_records(path, digest=first_digest):
            counts["records"] += 1
            kind = record["kind"]
            counts[kind] += 1
            if record["sequence"] != sequence + 1:
                integrity["sequence_gap"] += 1
            if record["timestamp_ns"] < timestamp:
                integrity["non_monotonic_envelope_timestamp"] += 1
            sequence, timestamp = record["sequence"], record["timestamp_ns"]
            if version is None:
                version = record["schema_version"]
            elif record["schema_version"] != version:
                integrity["mixed_trace_versions"] += 1
            if summary is not None:
                integrity["records_after_summary"] += 1
            if session is None and kind != "session":
                integrity["missing_initial_session"] += 1
            if kind == "session":
                if session is not None or sequence != 1:
                    integrity["multiple_or_misplaced_session"] += 1
                session = record
                if record["timestamp_clock"] != "steady_clock" or record["activity_clock"] != "cupti":
                    integrity["unknown_clock_domain"] += 1
            elif kind == "summary":
                summary = record
            elif kind == "gap":
                # Even a known diagnostic limitation cannot certify full coverage.
                integrity["collector_gap"] += 1
            elif kind == "api_enter":
                thread = record["thread_id"]
                if thread not in stacks:
                    stacks[thread] = []
                    stored += 1
                stack = stacks[thread]
                key = (record["domain"], record["correlation_id"])
                if (thread, key) in active:
                    issues["duplicate_active_api"] += 1
                active.add((thread, key))
                if len(stack) >= MAX_NESTING_DEPTH:
                    raise AuditInputError("correlation_nesting_limit")
                parent = None
                if record.get("op") == "launch":
                    if key in launches:
                        issues["reused_launch_correlation"] += 1
                    else:
                        if record["domain"] == "driver":
                            for enclosing in reversed(stack):
                                if enclosing["domain"] == "runtime" and enclosing.get("op") == "launch":
                                    parent = ("runtime", enclosing["correlation_id"])
                                    break
                        launches[key] = {"entry": record, "exit": None, "parent": parent}
                        stored += 1
                stack.append(record)
                stored += 1
            elif kind == "api_exit":
                key = (record["domain"], record["correlation_id"])
                stack = stacks.get(record["thread_id"], [])
                expected = (record["domain"], record["correlation_id"], record["symbol"])
                if not stack or (stack[-1]["domain"], stack[-1]["correlation_id"], stack[-1]["symbol"]) != expected:
                    issues["api_stack_mismatch"] += 1
                else:
                    entered = stack.pop()
                    active.discard((record["thread_id"], key))
                    stored -= 1
                    if entered.get("op") != record.get("op"):
                        issues["api_operation_changed"] += 1
                    if entered.get("op") == "launch":
                        launch = launches.get(key)
                        if launch is None or launch["entry"] is not entered or launch["exit"] is not None:
                            issues["duplicate_or_missing_launch_exit"] += 1
                        else:
                            launch["exit"] = record
                            if not entered.get("detail_known") or not record.get("detail_known"):
                                issues["unknown_launch_details"] += 1
                            if any(field not in entered or entered.get(field) != record.get(field) for field in _LAUNCH_FIELDS):
                                issues["launch_entry_exit_metadata_mismatch"] += 1
            elif kind == "activity":
                activity_kind = record["activity_kind"]
                if activity_kind == "kernel":
                    counts["gpu_kernel_activities"] += 1
                    if not record.get("correlation_id"):
                        issues["kernel_without_correlation"] += 1
                    else:
                        activities[record["correlation_id"]].append(record)
                        stored += 1
            if stored > MAX_STATE_ITEMS:
                raise AuditInputError("correlation_state_limit")
    except AuditInputError as error:
        integrity[str(error)] += 1
        input_valid = False

    # API activities carry no API name in v1 and may arrive before callbacks.
    # A bounded second pass stores only launch IDs, not every CUDA query in a run.
    if input_valid:
        try:
            for record in _trace_records(path, digest=second_digest):
                if record["kind"] == "activity" and record["activity_kind"] in ("api_driver", "api_runtime"):
                    key = (record["activity_kind"][4:], record.get("correlation_id", 0))
                    if key in launches:
                        api_activities[key] += 1
                        if not record.get("detail_known") or not record.get("start_ns") or not record.get("end_ns"):
                            issues["incomplete_api_activity"] += 1
                elif record["kind"] in ("api_enter", "api_exit"):
                    key = (record["domain"], record["correlation_id"])
                    if key in launches:
                        expected = launches[key]["entry" if record["kind"] == "api_enter" else "exit"]
                        if record != expected:
                            issues["conflicting_callback_correlation"] += 1
            if first_digest.digest() != second_digest.digest():
                integrity["trace_changed_between_passes"] += 1
                input_valid = False
        except AuditInputError as error:
            integrity[str(error)] += 1
            input_valid = False

    if any(stacks.values()):
        issues["unclosed_api_calls"] += sum(map(len, stacks.values()))
    if session is None:
        integrity["missing_session"] += 1
    if summary is None:
        integrity["missing_summary"] += 1
    else:
        if not summary["complete"] or not summary["safely_finalized"] or summary["terminal_checkpoint"] != "explicit_finalize":
            integrity["collector_not_safely_finalized"] += 1
        if summary["buffers_requested"] != summary["buffers_completed"]:
            integrity["uncompleted_activity_buffers"] += 1
        for field in _TERMINAL_ERRORS + ("unknown_callback_details",):
            if summary[field]:
                integrity[field] += summary[field]
        expected = {"callback_records": counts["api_enter"] + counts["api_exit"],
                    "activity_records": counts["activity"], "resource_records": counts["resource"]}
        if any(summary[field] != value for field, value in expected.items()):
            integrity["summary_counter_mismatch"] += 1

    successful = {}
    children = defaultdict(list)
    for key, launch in launches.items():
        domain = key[0]
        counts[domain + "_launch_callbacks"] += 1
        end = launch["exit"]
        if end is None:
            issues["launch_without_exit"] += 1
        elif end["status"] != 0:
            counts[domain + "_failed_launches"] += 1
            issues["failed_launch"] += 1
        else:
            successful[key] = launch
            counts[domain + "_successful_launches"] += 1
            if launch["parent"] is not None:
                children[launch["parent"]].append(key)
                counts["nested_driver_launches"] += 1
            elif domain == "driver":
                counts["direct_driver_launches"] += 1

    matched = set()
    for correlation, records in activities.items():
        if len(records) != 1:
            issues["duplicate_gpu_correlation"] += len(records)
            continue
        activity = records[0]
        candidates = [key for key in (("driver", correlation), ("runtime", correlation)) if key in successful]
        if len(candidates) == 2 and successful[candidates[0]]["parent"] == candidates[1]:
            # Same ID at two callback layers is only collapsed with observed nesting.
            candidates = candidates[:1]
        if len(candidates) != 1:
            issues["ambiguous_gpu_correlation" if candidates else "gpu_without_successful_launch"] += 1
            continue
        key = candidates[0]
        launch = successful[key]
        end = launch["exit"]
        valid = True
        if not activity.get("detail_known"):
            issues["unknown_gpu_activity_details"] += 1
            valid = False
        for field in ("context_id", "stream_id"):
            if not end.get(field) or not activity.get(field):
                issues["missing_" + field] += 1
                valid = False
            elif end[field] != activity[field]:
                issues[field + "_mismatch"] += 1
                valid = False
        if not end.get("kernel_name") or not activity.get("name") or end["kernel_name"] != activity["name"]:
            issues["kernel_name_mismatch_or_missing"] += 1
            valid = False
        for field in _GEOMETRY:
            if not end.get(field) or end.get(field) != activity.get(field):
                issues["geometry_mismatch_or_missing"] += 1
                valid = False
                break
        if "shared_bytes" not in end or end.get("shared_bytes") != activity.get("shared_bytes"):
            issues["shared_bytes_mismatch_or_missing"] += 1
            valid = False
        if not activity.get("start_ns") or not activity.get("end_ns"):
            issues["incomplete_kernel_timestamps"] += 1
            valid = False
        # CUPTI may emit only the outer Runtime API activity for a nested launch.
        # Accept that record only with an explicitly observed enclosing invocation.
        related = [key]
        if launch["parent"] is not None and launch["parent"] in successful:
            related.append(launch["parent"])
        if not any(api_activities[item] == 1 for item in related) or any(api_activities[item] > 1 for item in related):
            issues["api_activity_missing_or_duplicate"] += 1
            valid = False
        if valid:
            matched.add(key)
            counts["correlated_gpu_activities"] += 1
            parent = launch["parent"]
            if parent is not None:
                if parent not in successful:
                    issues["driver_inside_failed_or_unclosed_runtime"] += 1
                else:
                    outer = successful[parent]
                    for field in ("context_id", "stream_id", "kernel_name", "shared_bytes") + _GEOMETRY:
                        if outer["exit"].get(field) != end.get(field):
                            issues["runtime_driver_launch_metadata_mismatch"] += 1
                            break
                    if outer["entry"]["sequence"] < launch["entry"]["sequence"] < end["sequence"] < outer["exit"]["sequence"]:
                        counts["runtime_driver_gpu_chains"] += 1
                    else:
                        issues["invalid_nested_launch_interval"] += 1

    for key in successful:
        if key not in matched:
            if key[0] == "runtime" and len(children[key]) == 1 and children[key][0] in matched:
                continue
            issues["successful_launch_without_unique_gpu_activity"] += 1
    if not counts["gpu_kernel_activities"]:
        issues["empty_kernel_coverage"] += 1
    fields = ("records", "runtime_launch_callbacks", "driver_launch_callbacks",
              "runtime_successful_launches", "driver_successful_launches",
              "runtime_failed_launches", "driver_failed_launches", "nested_driver_launches",
              "direct_driver_launches", "gpu_kernel_activities", "correlated_gpu_activities",
              "runtime_driver_gpu_chains")
    return {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_correlations",
            **{field: counts[field] for field in fields},
            "launch_records_reconciled": input_valid and not issues and not any(code in integrity for code in (
                "sequence_gap", "mixed_trace_versions", "non_monotonic_envelope_timestamp",
                "multiple_or_misplaced_session", "missing_initial_session", "records_after_summary")),
            "trace_complete": not integrity,
            "issues": [{"code": code, "count": count} for code, count in sorted(issues.items())],
            "trace_issues": [{"code": code, "count": count} for code, count in sorted(integrity.items())],
            "device_ordering_proven": False, "semantic_ranges_proven": False,
            "execution_ready": False}
