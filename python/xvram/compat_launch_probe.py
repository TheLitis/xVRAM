"""Opt-in, pinned Windows CUDA Runtime routing/metadata experiment, not residency."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys

from .compat_audit import (VERSION, _Parser, build_command, clean_capture_environment,
                           load_profile, sha256_file, verify_binary_directory, verify_model)
from .compat_audit_capture import run_process

NATIVE_NAME = "xvram_native_cudart64_13.dll"
NATIVE_SHA256 = "b00ca6f53699120da815bf3e06e2e4285fae2f201235b883dcbb50eec51e2a2a"
WRAPPED = {"cudaLaunchKernel": "xvram_probe_launch", "cudaLaunchKernelExC": "xvram_probe_launch_ex"}
OBSERVED_APIS = set(WRAPPED) | {"cuLaunchKernel", "cuLaunchKernel_ptsz", "cuLaunchKernel_resolved_ptsz"}
TRACE_CAP = 128 * 1024 * 1024


def pe_exports(data: bytes):
    """Bounded PE32+ export metadata. No loading or executing the inspected image."""
    def read(offset, count):
        if offset < 0 or count < 0 or offset + count > len(data):
            raise ValueError("pe_bounds")
        return data[offset:offset + count]

    def unpack(fmt, offset):
        return struct.unpack(fmt, read(offset, struct.calcsize(fmt)))

    if len(data) > 64 * 1024 * 1024 or read(0, 2) != b"MZ":
        raise ValueError("pe_image")
    pe, = unpack("<I", 60)
    if read(pe, 4) != b"PE\0\0":
        raise ValueError("pe_signature")
    machine, count = unpack("<HH", pe + 4)
    optional_size, = unpack("<H", pe + 20)
    if machine != 0x8664 or not 1 <= count <= 96 or not 120 <= optional_size <= 4096:
        raise ValueError("pe_profile")
    optional = pe + 24
    if unpack("<H", optional)[0] != 0x20B or unpack("<I", optional + 108)[0] < 1:
        raise ValueError("pe_optional")
    export_rva, export_size = unpack("<II", optional + 112)
    if export_size < 40 or export_rva + export_size > 2**32:
        raise ValueError("pe_export_directory")
    sections = []
    for index in range(count):
        _, rva, raw_size, raw = unpack("<IIII", optional + optional_size + 40 * index + 8)
        read(raw, raw_size)
        if rva + raw_size > 2**32:
            raise ValueError("pe_section_overflow")
        sections.append((rva, raw_size, raw))

    def resolve(rva, size):
        matches = [raw + rva - base for base, length, raw in sections
                   if base <= rva and rva + size <= base + length]
        if len(matches) != 1:
            raise ValueError("pe_rva")
        return matches[0]

    def string(rva):
        value = bytearray()
        for delta in range(256):
            ch = read(resolve(rva + delta, 1), 1)
            if ch == b"\0":
                text = value.decode("ascii")
                if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", text):
                    raise ValueError("pe_export_name")
                return text
            value.extend(ch)
        raise ValueError("pe_export_name_limit")

    directory = resolve(export_rva, 40)
    base, functions, names, table, name_table, ordinals = unpack("<IIIIII", directory + 16)
    if not 1 <= names == functions <= 4096 or not 1 <= base or base + functions > 65536:
        raise ValueError("pe_export_count")
    result = []
    for index in range(names):
        name_rva, = unpack("<I", resolve(name_table + 4 * index, 4))
        ordinal, = unpack("<H", resolve(ordinals + 2 * index, 2))
        if ordinal >= functions:
            raise ValueError("pe_export_ordinal")
        target, = unpack("<I", resolve(table + 4 * ordinal, 4))
        if not target or export_rva <= target < export_rva + export_size:
            raise ValueError("pe_native_forwarder_or_hole")
        resolve(target, 1)
        result.append((string(name_rva), base + ordinal))
    if len({name for name, _ in result}) != names or len({ordinal for _, ordinal in result}) != names:
        raise ValueError("pe_export_duplicate")
    return sorted(result)


def export_definition(native: Path, *, import_definition=False):
    if native.is_symlink() or native.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("native_image_limit")
    with native.open("rb") as stream:
        data = stream.read(64 * 1024 * 1024 + 1)
    if hashlib.sha256(data).hexdigest() != NATIVE_SHA256:
        raise ValueError("native_hash_mismatch")
    exports = pe_exports(data)
    if not set(WRAPPED) <= {name for name, _ in exports}:
        raise ValueError("missing_wrapped_export")
    lines = ["LIBRARY " + (NATIVE_NAME if import_definition else "cudart64_13"), "EXPORTS"]
    for name, ordinal in exports:
        if import_definition:
            lines.append(f"  {name} @{ordinal}")
            continue
        target = WRAPPED.get(name, NATIVE_NAME[:-4] + "." + name)
        lines.append(f"  {name}={target} @{ordinal} PRIVATE")
    return "\n".join(lines) + "\n"


def stage(binary_dir: Path, proxy: Path, destination: Path):
    profile = load_profile()
    paths = verify_binary_directory(binary_dir, profile)
    if proxy.is_symlink() or not proxy.is_file() or proxy.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("invalid_proxy")
    proxy_hash = sha256_file(proxy)
    # No in-place operation, directory merging, or overwrite of original binaries.
    destination.mkdir(parents=True, exist_ok=False)
    expected = {}
    for path in paths:
        name = NATIVE_NAME if path.name == "cudart64_13.dll" else path.name
        expected[name] = profile["upstream"]["files"][path.name]
        shutil.copyfile(path, destination / name)
    expected["cudart64_13.dll"] = proxy_hash
    shutil.copyfile(proxy, destination / "cudart64_13.dll")
    manifest = {"schema_version": 1, "profile_id": profile["profile_id"], "files": expected}
    for name, digest in expected.items():
        if sha256_file(destination / name) != digest:
            raise ValueError("staged_copy_hash_mismatch")
    (destination / "launch-probe-stage.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_stage(directory: Path):
    path = directory / "launch-probe-stage.json"
    if path.stat().st_size > 64 * 1024:
        raise ValueError("stage_manifest_limit")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(load_profile()["upstream"]["files"])
    expected[NATIVE_NAME] = expected.pop("cudart64_13.dll")
    actual = manifest["files"]
    if set(actual) != set(expected) | {"cudart64_13.dll"} or any(actual[k] != v for k, v in expected.items()):
        raise ValueError("stage_profile_mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", actual["cudart64_13.dll"]):
        raise ValueError("stage_proxy_hash")
    if {p.name for p in directory.iterdir() if p.suffix.lower() in (".exe", ".dll")} != set(actual):
        raise ValueError("stage_binary_set")
    for name, digest in actual.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("stage_hash_mismatch")
    return actual


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def analyze_trace(path: Path):
    """Only observed host API pairs; deliberately cannot establish GPU coverage."""
    counts = {"records": 0, "calls_begun": 0, "calls_returned": 0, "metadata_calls": 0,
              "parameters": 0, "native_errors": 0}
    active = None
    definitions = {}
    digest = hashlib.sha256()
    total = 0
    setup = False
    mode = None
    version = None
    common = {"schema_version", "record_type", "sequence", "kind", "call_id"}
    def integer(record, key, low=0, high=2**64 - 1):
        value = record[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("trace_integer")
        return value
    with path.open("rb") as stream:
        while raw := stream.readline(65537):
            total += len(raw)
            if len(raw) > 65536 or not raw.endswith(b"\n") or total > TRACE_CAP:
                raise ValueError("trace_framing_or_capacity")
            digest.update(raw)
            record = json.loads(raw, object_pairs_hook=_object)
            counts["records"] += 1
            if record["schema_version"] not in (1, 2) or type(record["schema_version"]) is not int or record["record_type"] != "xvram.cuda_launch_probe":
                raise ValueError("trace_version")
            if version is not None and version != record["schema_version"]:
                raise ValueError("trace_version_changed")
            version = record["schema_version"]
            if integer(record, "sequence", 1) != counts["records"]:
                raise ValueError("trace_sequence")
            kind = record["kind"]
            if kind == "setup":
                if set(record) != common | {"hooks"} or counts["records"] != 1 or integer(record, "call_id") != 0 or integer(record, "hooks") != version + 1:
                    raise ValueError("trace_setup")
                setup = True
            elif kind == "begin":
                fields = common | {"api", "metadata", "kernel_name", "parameter_count", "module_resolved"}
                if version == 2:
                    fields |= {"handle_kind"}
                    if record.get("handle_kind") != ("contextless_kernel" if record.get("api") == "cuLaunchKernel_resolved_legacy" else "function"):
                        raise ValueError("trace_handle_kind")
                if set(record) != fields:
                    raise ValueError("trace_fields")
                if active is not None or integer(record, "call_id", 1) != counts["calls_begun"] + 1:
                    raise ValueError("trace_call_order")
                allowed = OBSERVED_APIS if version == 1 else {"cuLaunchKernel", "cuLaunchKernel_resolved_ptsz", "cuLaunchKernel_resolved_legacy"}
                if record["api"] not in allowed or type(record["metadata"]) is not bool or type(record["module_resolved"]) is not bool:
                    raise ValueError("trace_api")
                if record["api"].startswith("cuLaunch") != setup:
                    raise ValueError("trace_route_setup")
                if mode is not None and mode != record["metadata"]:
                    raise ValueError("trace_mode_changed")
                mode = record["metadata"]
                count = integer(record, "parameter_count", 0, 256)
                name = record["kernel_name"]
                if not isinstance(name, str) or len(name) > 4096 or re.search(r"0[xX][0-9a-fA-F]{6,}|(?<![\w])[0-9a-fA-F]{16}(?![\w])", name):
                    raise ValueError("trace_name_privacy")
                if record["metadata"] != record["module_resolved"] or (record["metadata"] and not name) or (not record["metadata"] and (name or count)):
                    raise ValueError("trace_metadata")
                active = {"call": record["call_id"], "name": name, "count": count, "parameters": []}
                counts["calls_begun"] += 1
                counts["metadata_calls"] += int(record["metadata"])
            elif kind == "parameter":
                if set(record) != common | {"index", "offset_bytes", "size_bytes"} or active is None or integer(record, "call_id", 1) != active["call"]:
                    raise ValueError("trace_parameter_order")
                params = active["parameters"]
                if integer(record, "index") != len(params) or len(params) >= active["count"]:
                    raise ValueError("trace_parameter_index")
                offset, size = integer(record, "offset_bytes", 0, 65536), integer(record, "size_bytes", 1, 65536)
                if offset + size > 65536 or (params and offset < sum(params[-1])):
                    raise ValueError("trace_parameter_range")
                params.append((offset, size))
                counts["parameters"] += 1
            elif kind == "end":
                if set(record) != common | {"result"} or active is None or integer(record, "call_id", 1) != active["call"] or len(active["parameters"]) != active["count"]:
                    raise ValueError("trace_end_order")
                counts["native_errors"] += int(integer(record, "result", 0, 2**31 - 1) != 0)
                counts["calls_returned"] += 1
                if active["name"]:
                    key = (active["name"], tuple(active["parameters"]))
                    definitions[key] = definitions.get(key, 0) + 1
                    if len(definitions) > 4096:
                        raise ValueError("trace_definition_limit")
                active = None
            else:
                raise ValueError("trace_kind")
    if active is not None or not counts["calls_returned"]:
        raise ValueError("trace_incomplete_or_empty")
    return {"sha256": digest.hexdigest(), "counts": counts, "driver_setup_observed": setup,
            "layouts": [{"kernel_name": name, "parameters": [{"offset_bytes": o, "size_bytes": s} for o, s in params],
                         "calls": calls} for (name, params), calls in sorted(definitions.items())]}


PROOF_FIELDS = {"transparent_execution", "oversubscription", "all_launches_covered", "cubin_binding",
                "semantic_ranges", "gpu_completion"}


def make_report(mode, mechanism, binaries, capture, trace_path, *, microbatch):
    capture_keys = {"exit_code", "timed_out", "controller_reaped", "process_tree_drained", "elapsed_ms",
                    "stdout_sha256", "output_truncated", "errors"}
    observed = {key: capture.get(key) for key in capture_keys}
    observed["output_truncated"] = capture.get("output_truncated", True)
    report = {"schema_version": 1, "report_type": "xvram.cuda_launch_probe", "version": VERSION,
              "mode": mode, "mechanism": mechanism, "binary_sha256": binaries, "capture": observed,
              "configuration": {"model": "14b", "microbatch": microbatch, "gpu_layers": 8,
                                "context_size": 2048, "generated_token_limit": 32},
              "trace": None, "diagnostics": [], "proof": dict.fromkeys(sorted(PROOF_FIELDS), False)}
    code = 26 if capture.get("timed_out") else 27
    try:
        report["trace"] = analyze_trace(trace_path)
        counters = report["trace"]["counts"]
        expected_metadata = counters["calls_returned"] if mode == "metadata" else 0
        if (capture["exit_code"] == 0 and not capture["errors"] and capture["controller_reaped"]
                and capture["process_tree_drained"] and not observed["output_truncated"]
                and not counters["native_errors"] and counters["metadata_calls"] == expected_metadata
                and report["trace"]["driver_setup_observed"] == (mechanism == "driver_entry_probe")):
            code = 0
    except (ValueError, KeyError, TypeError, OSError, UnicodeError):
        report["diagnostics"].append("trace_invalid_or_missing")
    report["exit_code"] = code
    validate_report(report)
    return report


def validate_report(report):
    """Dependency-free semantic checks; strict JSON Schema is the shape gate."""
    if (report.get("schema_version") != 1 or type(report["schema_version"]) is not int
            or report.get("report_type") != "xvram.cuda_launch_probe"
            or report.get("mode") not in ("routing", "metadata")
            or report.get("mechanism") not in ("runtime_proxy", "driver_entry_probe")):
        raise ValueError("report_profile")
    if set(report.get("proof", {})) != PROOF_FIELDS or any(v is not False for v in report["proof"].values()):
        raise ValueError("unsupported_proof_claim")
    if report.get("exit_code") not in (0, 23, 26, 27, 64, 70, 74) or type(report["exit_code"]) is not int:
        raise ValueError("report_exit")
    trace = report.get("trace")
    if trace is not None:
        counts = trace["counts"]
        if any(type(v) is not int or not 0 <= v <= 2**64-1 for v in counts.values()):
            raise ValueError("report_counter")
        if (counts["calls_begun"] != counts["calls_returned"] or counts["calls_returned"] == 0
                or counts["metadata_calls"] > counts["calls_returned"] or counts["native_errors"] > counts["calls_returned"]
                or counts["records"] != 2 * counts["calls_returned"] + counts["parameters"] + int(trace["driver_setup_observed"])):
            raise ValueError("report_reconciliation")
        if (sum(row["calls"] for row in trace["layouts"]) != counts["metadata_calls"]
                or sum(row["calls"] * len(row["parameters"]) for row in trace["layouts"]) != counts["parameters"]):
            raise ValueError("report_layout_reconciliation")
    if report["exit_code"] == 0:
        capture = report["capture"]
        if (trace is None or capture["exit_code"] != 0 or capture["timed_out"] is not False
                or capture["controller_reaped"] is not True or capture["process_tree_drained"] is not True
                or capture["output_truncated"] is not False or capture["errors"] or report["diagnostics"]
                or trace["counts"]["native_errors"] != 0
                or trace["driver_setup_observed"] != (report["mechanism"] == "driver_entry_probe")
                or trace["counts"]["metadata_calls"] != (trace["counts"]["calls_returned"] if report["mode"] == "metadata" else 0)):
            raise ValueError("false_completed_report")
    if report["capture"]["timed_out"] is True and report["exit_code"] != 26:
        raise ValueError("timeout_exit_mismatch")


def main(argv=None):
    parser = _Parser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    exports = sub.add_parser("exports")
    exports.add_argument("--native", type=Path, required=True)
    exports.add_argument("--output", type=Path, required=True)
    exports.add_argument("--import-definition", action="store_true")
    staging = sub.add_parser("stage")
    staging.add_argument("--binary-dir", type=Path, required=True)
    staging.add_argument("--proxy", type=Path, required=True)
    staging.add_argument("--destination", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--binary-dir", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--mode", choices=("routing", "metadata"), required=True)
    run.add_argument("--driver-probe", type=Path,
                     help="Use the active Driver probe with the original pinned binary directory")
    run.add_argument("--census-cupti-dir", type=Path,
                     help="Enable simultaneous census using a census-built Driver DLL and pinned CUPTI directory")
    run.add_argument("--identity-witness", action="store_true",
                     help="Collect a separate identity sidecar with a witness-enabled diagnostic DLL")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--microbatch", type=int, choices=(1, 128), default=128)
    run.add_argument("--timeout-seconds", type=int, default=900, help="total native-run deadline, 1..900 seconds")
    args = parser.parse_args(argv)
    if args.action == "run" and not 1 <= args.timeout_seconds <= 900:
        parser.error("timeout must be 1..900 seconds")
    if args.action == "run" and args.census_cupti_dir and not args.driver_probe:
        parser.error("census requires the active Driver probe")
    if args.action == "run" and args.identity_witness and (not args.census_cupti_dir or args.mode != "metadata"):
        parser.error("identity witness requires census and metadata mode")
    try:
        if args.action == "exports":
            definition = export_definition(args.native, import_definition=args.import_definition)
            with args.output.open("x", encoding="ascii") as stream:
                stream.write(definition)
            return 0
        if args.action == "stage":
            stage(args.binary_dir, args.proxy, args.destination)
            return 0
        if os.name != "nt":
            raise ValueError("windows_only_native_profile")
        if args.driver_probe:
            original = verify_binary_directory(args.binary_dir, load_profile())
            binaries = {path.name: sha256_file(path) for path in original}
            if args.driver_probe.is_symlink() or not args.driver_probe.is_file():
                raise ValueError("driver_probe_unavailable")
            binaries["xvram_driver_launch_probe.dll"] = sha256_file(args.driver_probe)
        else:
            binaries = verify_stage(args.binary_dir)
        if args.census_cupti_dir:
            from .compat_launch_census import CUPTI_NAME, CUPTI_SHA256
            if sha256_file(args.census_cupti_dir / CUPTI_NAME) != CUPTI_SHA256:
                raise ValueError("census_cupti_pin")
        profile = load_profile()
        shards = verify_model(args.model_dir, profile["models"]["14b"])
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        args.application, args.context_size, args.gpu_layers = "completion", 2048, 8
        args.generate, args.prompt_file = 32, None
        environment = clean_capture_environment()
        environment["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        environment["XVRAM_LAUNCH_PROBE_MODE"] = args.mode
        environment["XVRAM_LAUNCH_PROBE_TRACE"] = str(args.output_dir / "launches.jsonl")
        if args.identity_witness:
            environment["XVRAM_IDENTITY_WITNESS_TRACE"] = str(args.output_dir / "identity.jsonl")
        if args.driver_probe:
            environment["CUDA_INJECTION64_PATH"] = str(args.driver_probe.resolve())
        if args.census_cupti_dir:
            environment["XVRAM_LAUNCH_CENSUS_TRACE"] = str(args.output_dir / "census.jsonl")
            environment["PATH"] = str(args.census_cupti_dir.resolve()) + os.pathsep + environment.get("PATH", "")
        environment["PATH"] = str(args.binary_dir.resolve()) + os.pathsep + environment.get("PATH", "")
        result = run_process(build_command(args, shards[0]), output_dir=args.output_dir,
                             timeout_seconds=args.timeout_seconds, environment=environment,
                             cwd=args.binary_dir, sample_gpu=True)
        report = make_report(args.mode, "driver_entry_probe" if args.driver_probe else "runtime_proxy",
                             binaries, result, args.output_dir / "launches.jsonl", microbatch=args.microbatch)
        (args.output_dir / "probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.census_cupti_dir:
            from .compat_launch_census import make_report as census_report
            census = census_report(report, args.output_dir / "census.jsonl")
            (args.output_dir / "census-report.json").write_text(json.dumps(census, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if args.identity_witness:
                from .compat_audit_identity import make_report as identity_report
                witness = identity_report(report, census, args.output_dir / "identity.jsonl", args.output_dir / "census.jsonl")
                (args.output_dir / "identity-report.json").write_text(json.dumps(witness, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                return witness["exit_code"]
            return census["exit_code"]
        return report["exit_code"]
    except (ValueError, KeyError, TypeError, OSError, UnicodeError) as error:
        print(f"launch probe preflight/output failure: {type(error).__name__}", file=sys.stderr)
        return 23 if not isinstance(error, OSError) else 74


if __name__ == "__main__":
    raise SystemExit(main())
