"""Read-only, bounded summaries of official Nsight Systems SQLite exports.

This is diagnostic aggregation, not a normalized residency trace or GO evidence.
Only fixed timing/name/transfer fields are read; native addresses and handles are
never selected. No profiler artifact is modified or deleted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

MAX_DATABASE_BYTES = 2 * 1024**3
MAX_TABLE_ROWS = 500_000
MAX_STRING_ROWS = 100_000
MAX_DISTINCT_NAMES = 4096
MAX_NAME_CHARS = 2048
MAX_QUERY_SECONDS = 15
MAX_ENUM_ROWS = 256
MAX_DIAGNOSTIC_ROWS = 4096
MAX_DIAGNOSTIC_CHARS = 4096
_UINT64_MAX = (1 << 64) - 1
_TABLES = {
    "kernels": "CUPTI_ACTIVITY_KIND_KERNEL",
    "runtime": "CUPTI_ACTIVITY_KIND_RUNTIME",
    "driver": "CUPTI_ACTIVITY_KIND_DRIVER",
    "memcpy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
}
_HEX = re.compile(r"(?i)\b0x[0-9a-f]+\b|\b[0-9a-f]{12,16}\b")
_HANDLE = re.compile(r"(?i)\b(?:stream|context|module|handle|pointer|address)\s*[:=]\s*[^\s,;]+")
_API_CLASSES = {"TRACE_PROCESS_EVENT_CUDA_RUNTIME": "runtime",
                "TRACE_PROCESS_EVENT_CUDA_DRIVER": "driver",
                "TRACE_PROCESS_EVENT_CUBLAS": "cublas",
                "TRACE_PROCESS_EVENT_CUDNN": "cudnn"}
_SEVERITIES = {"Info": "info", "Warning": "warning", "Error": "error", "Verbose": "verbose"}


class _SummaryLimit(ValueError):
    pass


def _unresolved(summary: dict[str, Any], code: str) -> None:
    if code not in summary["unresolved"]:
        summary["unresolved"].append(code)


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise _SummaryLimit("normalization_deadline_exceeded")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _columns(connection: sqlite3.Connection, table: str) -> dict[str, str] | None:
    definition = connection.execute(
        "SELECT type, substr(sql, 1, 128) FROM sqlite_master WHERE name=?", (table,)
    ).fetchone()
    if definition is None:
        return None
    if definition[0] != "table" or not isinstance(definition[1], str) or not re.match(
            r"\s*CREATE\s+TABLE\b", definition[1], flags=re.IGNORECASE):
        raise _SummaryLimit("unsupported_table_kind")
    rows = connection.execute("PRAGMA table_info(" + _quote_identifier(table) + ")").fetchmany(129)
    if len(rows) > 128:
        raise _SummaryLimit("column_limit_exceeded")
    return {row[1].casefold(): row[1] for row in rows}


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _UINT64_MAX


def _sum(total: int | None, value: int) -> int | None:
    if total is None or total > _UINT64_MAX - value:
        return None
    return total + value


def _aggregate() -> dict[str, Any]:
    return {"count": 0, "timed_count": 0, "invalid_duration_count": 0,
            "total_duration_ns": 0, "min_duration_ns": None, "max_duration_ns": None}


def _add_interval(target: dict[str, Any], start: Any, end: Any) -> None:
    target["count"] += 1
    if not _integer(start) or not _integer(end) or end < start:
        target["invalid_duration_count"] += 1
        return
    duration = end - start
    target["timed_count"] += 1
    target["total_duration_ns"] = _sum(target["total_duration_ns"], duration)
    minimum, maximum = target["min_duration_ns"], target["max_duration_ns"]
    target["min_duration_ns"] = duration if minimum is None else min(minimum, duration)
    target["max_duration_ns"] = duration if maximum is None else max(maximum, duration)


def _name(value: Any) -> str:
    if not isinstance(value, str):
        return "[name-unavailable]"
    value = _HANDLE.sub("[native-value-redacted]", _HEX.sub("[native-value-redacted]", value))
    return "".join(character if character.isprintable() else " " for character in value)


def _read_enumeration(connection: sqlite3.Connection, table: str, allowed: dict[str, str],
                      default: str, deadline: float) -> dict[int, str]:
    columns = _columns(connection, table)
    if columns is None or not {"id", "name"}.issubset(columns):
        return {}
    query = "SELECT " + _quote_identifier(columns["id"]) + ", substr(" + _quote_identifier(columns["name"]) + ",1,257) FROM " + _quote_identifier(table) + " LIMIT ?"
    result: dict[int, str] = {}
    for index, (identity, name) in enumerate(connection.execute(query, (MAX_ENUM_ROWS + 1,))):
        _deadline(deadline)
        if index >= MAX_ENUM_ROWS:
            raise _SummaryLimit("enumeration_row_limit_exceeded")
        if not _integer(identity) or not isinstance(name, str) or len(name) > 256:
            raise _SummaryLimit("enumeration_row_invalid")
        if identity in result:
            raise _SummaryLimit("enumeration_identity_ambiguous")
        # Untrusted enum names never become output strings.
        result[identity] = allowed.get(name, default)
    return result


def _read_diagnostics(connection: sqlite3.Connection, summary: dict[str, Any], deadline: float) -> None:
    columns = _columns(connection, "DIAGNOSTIC_EVENT")
    if columns is None or not {"severity", "text"}.issubset(columns):
        _unresolved(summary, "profiler_diagnostics_unavailable")
        return
    severities = _read_enumeration(connection, "ENUM_DIAGNOSTIC_SEVERITY_LEVEL", _SEVERITIES, "unknown", deadline)
    observation = summary["profiler_diagnostics"]
    observation.update(status="observed", count=0,
                       severity_counts={level: 0 for level in ("info", "warning", "error", "verbose", "unknown")})
    query = "SELECT " + _quote_identifier(columns["severity"]) + ", substr(" + _quote_identifier(columns["text"]) + ",1,?) FROM DIAGNOSTIC_EVENT LIMIT ?"
    codes: set[str] = set()
    for index, (severity, message) in enumerate(connection.execute(query, (MAX_DIAGNOSTIC_CHARS + 1, MAX_DIAGNOSTIC_ROWS + 1))):
        _deadline(deadline)
        if index >= MAX_DIAGNOSTIC_ROWS:
            observation["status"] = "truncated"
            _unresolved(summary, "profiler_diagnostic_row_limit_exceeded")
            break
        observation["count"] += 1
        level = severities.get(severity, "unknown") if _integer(severity) else "unknown"
        observation["severity_counts"][level] += 1
        if not isinstance(message, str) or len(message) > MAX_DIAGNOSTIC_CHARS:
            _unresolved(summary, "profiler_diagnostic_text_invalid_or_truncated")
            continue
        lower = message.casefold()
        # Retain only fixed semantic codes, never raw text (paths/PIDs/pointers).
        if "installed cuda driver version" in lower and "not supported by this build of nsight systems" in lower:
            codes.add("driver_version_outside_profiler_support")
            _unresolved(summary, "profiler_driver_version_outside_support")
        if "legacy (software instrumented) trace" in lower:
            codes.add("legacy_software_instrumentation")
        if message.strip() == "Profiling has started.":
            codes.add("profiling_start_observed")
        if message.strip() == "Profiling has stopped.":
            codes.add("profiling_stop_observed")
    observation["codes"] = sorted(codes)
    for level in ("warning", "error", "unknown"):
        if observation["severity_counts"][level]:
            _unresolved(summary, "profiler_diagnostic_" + level + "_observed")


def _read_activity(connection: sqlite3.Connection, summary: dict[str, Any], kind: str,
                   deadline: float, api_classes: dict[int, str]) -> dict[Any, dict[str, Any]]:
    table = _TABLES[kind]
    columns = _columns(connection, table)
    if columns is None:
        summary["tables"][kind] = "missing"
        _unresolved(summary, kind + "_table_unobserved")
        return {}
    if not {"start", "end"}.issubset(columns):
        summary["tables"][kind] = "unsupported_columns"
        _unresolved(summary, kind + "_timing_columns_unavailable")
        return {}
    names = ("demangledname", "mangledname", "shortname", "nameid", "name") if kind == "kernels" else ("nameid", "name")
    name_columns = [columns[name] for name in names if name in columns]
    memory = kind in ("memcpy", "memset")
    if not memory and not name_columns:
        summary["tables"][kind] = "unsupported_columns"
        _unresolved(summary, kind + "_name_column_unavailable")
        return {}
    selected = [_quote_identifier(columns[key]) for key in ("start", "end")]
    name_expression = ",".join(_quote_identifier(column) for column in name_columns)
    if len(name_columns) > 1:
        name_expression = "coalesce(" + name_expression + ")"
    selected.append(_quote_identifier(columns["bytes"]) if memory and "bytes" in columns
                    else ("NULL" if memory else name_expression))
    selected.append(_quote_identifier(columns["eventclass"]) if kind in ("runtime", "driver")
                    and "eventclass" in columns else "NULL")
    query = "SELECT " + ",".join(selected) + " FROM " + _quote_identifier(table)
    if kind == "runtime" and "callchainid" in columns:
        # Match the official cuda_api_sum report's exclusion of callchain rows.
        query += " WHERE " + _quote_identifier(columns["callchainid"]) + " IS NULL"
    query += " LIMIT ?"
    aggregates: dict[Any, dict[str, Any]] = {}
    summary["tables"][kind] = "observed"
    summary["counts"][kind] = 0
    if kind in ("runtime", "driver"):
        summary["api_class_counts"][kind] = {family: 0 for family in ("runtime", "driver", "cublas", "cudnn", "other", "unknown")}
    for index, (start, end, extra, event_class) in enumerate(connection.execute(query, (MAX_TABLE_ROWS + 1,))):
        _deadline(deadline)
        if index >= MAX_TABLE_ROWS:
            summary["tables"][kind] = "truncated"
            _unresolved(summary, kind + "_row_limit_exceeded")
            break
        summary["counts"][kind] += 1
        if memory:
            key = kind
        elif isinstance(extra, str):
            if len(extra) > MAX_NAME_CHARS:
                key = "[name-too-long]"
                _unresolved(summary, "name_length_limit_exceeded")
            else:
                key = extra
        elif _integer(extra):
            key = extra
        else:
            key = "[name-unavailable]"
            _unresolved(summary, "name_reference_invalid")
        if key not in aggregates and len(aggregates) >= MAX_DISTINCT_NAMES:
            summary["tables"][kind] = "truncated"
            _unresolved(summary, kind + "_distinct_name_limit_exceeded")
            # Do not count a row that cannot be reconciled to its bounded aggregate.
            summary["counts"][kind] -= 1
            break
        if key not in aggregates:
            aggregates[key] = _aggregate()
            if memory:
                aggregates[key].update(known_byte_count=0, invalid_byte_count=0, total_bytes=0)
        bucket = aggregates[key]
        _add_interval(bucket, start, end)
        if kind in ("runtime", "driver"):
            family = api_classes.get(event_class, "unknown") if _integer(event_class) else "unknown"
            summary["api_class_counts"][kind][family] += 1
        if memory:
            if _integer(extra):
                bucket["known_byte_count"] += 1
                bucket["total_bytes"] = _sum(bucket["total_bytes"], extra)
            else:
                bucket["invalid_byte_count"] += 1
                _unresolved(summary, kind + "_byte_observation_unavailable")
        if bucket["invalid_duration_count"]:
            _unresolved(summary, kind + "_invalid_duration")
        if bucket["total_duration_ns"] is None or (memory and bucket["total_bytes"] is None):
            _unresolved(summary, kind + "_aggregate_overflow")
    return aggregates


def _read_names(connection: sqlite3.Connection, needed: set[int], summary: dict[str, Any],
                deadline: float) -> dict[int, str]:
    if not needed:
        return {}
    columns = _columns(connection, "StringIds")
    if columns is None or not {"id", "value"}.issubset(columns):
        _unresolved(summary, "string_ids_unavailable")
        return {}
    query = "SELECT " + _quote_identifier(columns["id"]) + ", substr(" + _quote_identifier(columns["value"]) + ",1,?) FROM StringIds LIMIT ?"
    found: dict[int, str] = {}
    for index, (identity, value) in enumerate(connection.execute(query, (MAX_NAME_CHARS + 1, MAX_STRING_ROWS + 1))):
        _deadline(deadline)
        if index >= MAX_STRING_ROWS:
            _unresolved(summary, "string_row_limit_exceeded")
            break
        if not _integer(identity) or identity not in needed:
            continue
        if not isinstance(value, str) or len(value) > MAX_NAME_CHARS:
            _unresolved(summary, "string_name_invalid_or_truncated")
            continue
        if identity in found:
            _unresolved(summary, "duplicate_string_identity")
            found[identity] = "[ambiguous-name]"
        else:
            found[identity] = _name(value)
    if needed - found.keys():
        _unresolved(summary, "referenced_names_unobserved")
    return found


def summarize_nsys(path: Path) -> dict[str, Any]:
    """Aggregate a finalized SQLite export without granting readiness or modifying it."""
    summary: dict[str, Any] = {
        "schema_version": 1, "report_type": "xvram.cuda_compat_audit_nsys",
        "source": {"bytes": None, "sha256": None},
        "coverage_complete": False,
        "timing_scope": "sum_of_activity_intervals_not_application_wall_time",
        "api_scope": "runtime_callchain_rows_excluded_when_present_no_cross_domain_deduplication",
        "api_grouping": "physical_activity_tables_not_api_families_runtime_table_can_include_driver_calls",
        "memory_scope": "observed_transfer_bytes_not_resident_memory_or_wddm_budget",
        "limits": {"database_bytes": MAX_DATABASE_BYTES, "rows_per_table": MAX_TABLE_ROWS,
                   "string_rows": MAX_STRING_ROWS, "distinct_names_per_table": MAX_DISTINCT_NAMES,
                   "query_seconds": MAX_QUERY_SECONDS, "enumeration_rows": MAX_ENUM_ROWS,
                   "diagnostic_rows": MAX_DIAGNOSTIC_ROWS, "diagnostic_text_chars": MAX_DIAGNOSTIC_CHARS},
        "tables": {kind: "unobserved" for kind in _TABLES},
        "counts": {kind: None for kind in _TABLES},
        "kernels": [], "apis": {"runtime": [], "driver": [], "cublas": []}, "memory": [],
        "api_class_counts": {"runtime": None, "driver": None},
        "profiler_diagnostics": {"status": "unobserved", "count": None, "severity_counts": None, "codes": []},
        "unresolved": ["aggregate_only_not_launch_range_evidence", "event_lifetimes_not_reconstructed",
                       "cuBLAS_api_visibility_not_established", "profiler_drop_and_teardown_coverage_unverified"],
    }
    deadline = time.monotonic() + MAX_QUERY_SECONDS
    connection = None
    try:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise _SummaryLimit("database_unavailable")
        before = path.stat()
        if not 16 <= before.st_size <= MAX_DATABASE_BYTES:
            raise _SummaryLimit("database_size_outside_bound")
        summary["source"]["bytes"] = before.st_size
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(path) + suffix)
            if sidecar.exists() and sidecar.stat().st_size:
                raise _SummaryLimit("database_not_finalized_sidecar_present")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            header = stream.read(16)
            if header != b"SQLite format 3\0":
                raise _SummaryLimit("database_header_invalid")
            digest.update(header)
            while chunk := stream.read(1024 * 1024):
                _deadline(deadline)
                digest.update(chunk)
        summary["source"]["sha256"] = digest.hexdigest()
        # The caller supplies a completed, quiescent export. Reject live sidecars
        # above and detect size/mtime changes below instead of creating WAL/SHM.
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=1)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        if hasattr(connection, "enable_load_extension"):
            connection.enable_load_extension(False)
        if hasattr(connection, "setlimit"):
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 256 * 1024)
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            api_classes = _read_enumeration(connection, "ENUM_NSYS_EVENT_CLASS", _API_CLASSES, "other", deadline)
            if not api_classes:
                _unresolved(summary, "api_class_mapping_unavailable")
        except (_SummaryLimit, sqlite3.Error):
            api_classes = {}
            _unresolved(summary, "api_class_mapping_failed_or_query_deadline_exceeded")
        activities = {}
        for kind in _TABLES:
            try:
                activities[kind] = _read_activity(connection, summary, kind, deadline, api_classes)
            except (_SummaryLimit, sqlite3.Error) as error:
                activities[kind] = {}
                summary["tables"][kind] = "failed"
                summary["counts"][kind] = None
                if kind in ("runtime", "driver"):
                    summary["api_class_counts"][kind] = None
                _unresolved(summary, str(error) if isinstance(error, _SummaryLimit)
                            else kind + "_sqlite_read_failed_or_query_deadline_exceeded")
        try:
            _read_diagnostics(connection, summary, deadline)
        except (_SummaryLimit, sqlite3.Error):
            summary["profiler_diagnostics"] = {"status": "failed", "count": None, "severity_counts": None, "codes": []}
            _unresolved(summary, "profiler_diagnostics_failed_or_query_deadline_exceeded")
        needed = {key for kind in ("kernels", "runtime", "driver") for key in activities[kind]
                  if isinstance(key, int)}
        try:
            names = _read_names(connection, needed, summary, deadline)
        except (_SummaryLimit, sqlite3.Error) as error:
            names = {}
            _unresolved(summary, str(error) if isinstance(error, _SummaryLimit)
                        else "string_ids_read_failed_or_query_deadline_exceeded")
        for kind in ("kernels", "runtime", "driver"):
            destination = summary["kernels"] if kind == "kernels" else summary["apis"][kind]
            for key, bucket in activities[kind].items():
                name = names.get(key, "[name-unavailable]") if isinstance(key, int) else _name(key)
                if not bucket["timed_count"]:
                    bucket["total_duration_ns"] = None
                destination.append({"name": name, "name_sha256": hashlib.sha256(name.encode()).hexdigest(), **bucket})
            destination.sort(key=lambda item: (item["name"], item["count"], item["total_duration_ns"] or 0))
        for kind in ("memcpy", "memset"):
            for bucket in activities[kind].values():
                if not bucket["timed_count"]:
                    bucket["total_duration_ns"] = None
                if not bucket["known_byte_count"]:
                    bucket["total_bytes"] = None
                summary["memory"].append({"operation": kind, **bucket})
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            _unresolved(summary, "database_changed_during_observation")
    except _SummaryLimit as error:
        _unresolved(summary, str(error))
    except sqlite3.Error:
        _unresolved(summary, "sqlite_read_failed_or_query_deadline_exceeded")
    except (OSError, ValueError, TypeError):
        _unresolved(summary, "database_observation_failed")
    finally:
        if connection is not None:
            connection.close()
    return summary
