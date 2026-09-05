"""Strict no-driver and hardware contracts for Phase 6a; never alters old contracts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import tempfile

from jsonschema import Draft202012Validator


def validator(path: Path) -> Draft202012Validator:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def privacy(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in {"pointer", "address", "cuda_va", "native_handle", "native_stream", "stream_handle", "virtual_address"}, key
            privacy(item)
    elif isinstance(value, list):
        for item in value:
            privacy(item)
    elif isinstance(value, str):
        assert not re.search(r"\b0x[0-9a-fA-F]{9,}\b", value), "raw native address in output"


def validate_semantics(report: dict, *, fixture: bool = False) -> None:
    privacy(report)
    config, cache, execution, compat = (report[k] for k in ("configuration", "cache", "execution", "compatibility"))
    if not config["identifiers_included"]:
        assert report["device"]["uuid"] is None and report["device"]["pci_bus_id"] is None
    if report["outcome"]["exit_code"] != 0:
        assert report["outcome"]["status"] != "completed"
        if report["outcome"]["exit_code"] in (26, 74):
            assert not all(value is True for value in report["proof"].values())
        return
    assert all(value is True for value in report["proof"].values())
    assert all(value is True for value in report["cleanup"].values())
    assert report["diagnostics"] == []
    assert report["verification"]["mismatches"] == 0
    assert cache["mappings"] == cache["set_access_calls"] == cache["unmaps"]
    assert cache["unsafe_remaps"] == cache["unsafe_transitions"] == 0
    assert cache["resident_bytes"] == 0
    assert cache["resident_bytes_peak"] <= cache["cache_target_maximum_bytes"]
    assert cache["physical_handles_created"] == cache["physical_handles_released"]
    assert execution["tiles_submitted"] == execution["tiles_retired"]
    assert compat["live_allocations"] == compat["live_handles"] == 0
    assert compat["allocations_created"] == compat["allocations_released"]
    assert compat["handles_created"] == compat["handles_destroyed"]
    assert compat["retired_va_reservations"] == compat["retired_va_bytes"] == 0
    assert compat["rejections_failed"] == 0
    assert compat["calls_attempted"] == compat["calls_completed"] + compat["calls_rejected"]
    assert config["host_budget_bytes"] <= config["host_store_cap_bytes"]
    assert cache["pinned_staging_bytes"] <= config["effective_chunk_bytes"] * config["staging_slots"]
    workloads = report["workloads"]
    assert sum(w["tiles_retired"] for w in workloads) == execution["tiles_retired"]
    assert sum(w["mappings"] for w in workloads) == cache["mappings"]
    assert sum(w["unmaps"] for w in workloads) == cache["unmaps"]
    assert sum(w["passes_completed"] for w in workloads) == execution["gemm_calls"]
    if not fixture:
        assert sum(w["h2d_bytes"] for w in workloads) == cache["bytes_h2d"]
        assert sum(w["d2h_bytes"] for w in workloads) == cache["bytes_d2h"]
        assert cache["event_boundaries"] >= execution["tiles_retired"]
    for work in workloads:
        assert work["status"] == "completed" and work["mismatches"] == 0
        assert work["output_elements_checked"] == work["m"] * work["n"]
        assert work["logical_bytes"] == 4 * (work["m"] * work["k"] + work["k"] * work["n"] + work["m"] * work["n"])
        assert work["storage_bytes"] >= work["logical_bytes"]
        assert work["passes_completed"] == config["passes"]
        assert work["pass_timings"]["sample_count"] == config["passes"]
        assert work["mismatch_offset"] is None
        assert len(work["digest"]) == len(work["reference_digest"]) == 32
        if work["reference_kind"] == "cpu_fp64_full":
            assert work["native_baseline_equal"] is True
            assert work["native_baseline_ms"] is not None
        if work["logical_bytes"] > report["device"]["total_memory_bytes"]:
            assert work["evictions"] > 0 and work["handle_reuses"] > 0
    if config["scenario"] in ("suite", "rejection") and not fixture:
        assert compat["rejections_checked"] >= 8


def validate_trace(path: Path, trace_schema: Draft202012Validator, report: dict) -> None:
    sequence = 0
    timestamp = 0
    retired = 0
    active: dict[int, dict] = {}
    returned: set[int] = set()
    final = False
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        trace_schema.validate(record)
        privacy(record)
        assert not final, "record after terminal trace record"
        assert record["sequence"] == sequence + 1
        assert record["monotonic_timestamp_ns"] >= timestamp
        sequence, timestamp = record["sequence"], record["monotonic_timestamp_ns"]
        operation = record["operation_id"]
        if record["transition"] == "call":
            assert operation > 0 and operation not in active and operation not in returned
            assert not active, "synchronous profile overlapped public calls"
            active[operation] = record
        elif record["transition"] == "return":
            call = active.pop(operation)
            assert call["allocation_id"] == record["allocation_id"]
            assert call["bytes"] == record["bytes"] and call["reason"] == record["reason"]
            returned.add(operation)
        elif record["transition"] == "retired_tile":
            assert operation in active and active[operation]["reason"] == "cublasSgemm"
            assert record["tiles_retired"] == retired + 1
            retired += 1
        else:
            final = True
            if report["outcome"]["exit_code"] == 0:
                assert not active
        assert record["tiles_retired"] >= retired
    if report["outcome"]["exit_code"] == 0:
        assert final and not active
        assert report["execution"]["trace_complete"]
        assert report["execution"]["trace_records"] == sequence
        assert retired == report["execution"]["tiles_retired"]


def no_driver(bench: Path, schema: Draft202012Validator) -> None:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(bench), *args], capture_output=True, text=True, timeout=15, check=False)

    assert run("--help").returncode == 0
    assert run("--version").returncode == 0
    invalid = [
        ["--m", "2"], ["--logical-size", "1GiB", "--m", "2", "--n", "3", "--k", "4"],
        ["--op-a", "c"], ["--alpha", "nan"], ["--beta", "inf"], ["--passes", "0"],
        ["--policy", "both"], ["--staging-slots", "1"], ["--prefetch-distance", "9"],
        ["--timeout-seconds", "0"], ["--trace", "-"], ["--no-text"], ["--help", "--json", "-"],
    ]
    for args in invalid:
        result = run(*args)
        assert result.returncode == 64, (args, result.returncode, result.stderr)
    for mode, code in (("success", 23), ("crash", 27), ("truncated", 27), ("oversized", 27), ("non-monotonic", 27), ("stale-progress", 27), ("final-exit-mismatch", 27), ("contradictory-report", 27), ("malformed-json", 27), ("duplicate-json-key", 27), ("hang", 26), ("heartbeat-hang", 26)):
        result = run("--test-worker", mode, "--timeout-seconds", "1", "--json", "-", "--no-text", "--compact-json")
        assert result.returncode == code, (mode, result.returncode, result.stdout, result.stderr)
        report = json.loads(result.stdout)
        schema.validate(report)
        validate_semantics(report)
        assert report["outcome"]["exit_code"] == code
        assert report["cleanup"]["worker_terminated"] is True
        assert report["cleanup"]["adapter_closed"] is None
    pretty = run("--test-worker", "success", "--json", "-", "--no-text")
    assert pretty.returncode == 23 and "\n  " in pretty.stdout
    with tempfile.TemporaryDirectory(prefix="xvram-compat-contract-") as temporary:
        # A directory is a portable unwritable trace target. Controller must not launch work.
        result = run("--test-worker", "trace", "--trace", temporary, "--json", "-", "--no-text")
        assert result.returncode == 74
        report = json.loads(result.stdout)
        schema.validate(report)
        assert report["cleanup"]["trace_closed"] is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--trace-schema", type=Path, required=True)
    parser.add_argument("--fixture-helper", type=Path)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    schema, trace_schema = validator(args.schema), validator(args.trace_schema)
    embedded_path = args.schema.parent.parent / "src/compat_bench/report_schema_v1.hpp"
    embedded = embedded_path.read_text(encoding="utf-8").split('R"XVRAM(', 1)[1].split(')XVRAM"', 1)[0]
    assert json.loads(embedded) == schema.schema, "embedded controller schema drifted"
    if args.fixture_helper:
        fixtures = subprocess.run([str(args.fixture_helper), "--emit-fixtures"], check=True, text=True, capture_output=True)
        reports = [json.loads(line) for line in fixtures.stdout.splitlines()]
        assert len(reports) == 8
        for report in reports:
            schema.validate(report)
            validate_semantics(report, fixture=True)
        completed = reports[0]
        for section, key, value in (("proof", "event_safe", False), ("cleanup", "worker_terminated", None), ("verification", "mismatches", 1)):
            malformed = copy.deepcopy(completed)
            malformed[section][key] = value
            assert not schema.is_valid(malformed), (section, key)
        malformed = copy.deepcopy(completed)
        malformed["device"]["pointer"] = 123
        assert not schema.is_valid(malformed)
    if args.bench:
        no_driver(args.bench, schema)
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8-sig"))
        schema.validate(report)
        validate_semantics(report)
        if args.trace:
            validate_trace(args.trace, trace_schema, report)
    print("CUDA compatibility contracts passed")


if __name__ == "__main__":
    main()
