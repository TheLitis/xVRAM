"""Isolated PC-sampling permission gate; never changes NVIDIA/system settings."""
import argparse
import json
import os
from pathlib import Path
import sys

from .compat_audit import clean_capture_environment, sha256_file
from .compat_audit_capture import run_process
from .compat_launch_probe import _object

PINS = {
    "cupti64_2026.2.1.dll": "9b10d2fafaff1a4dc9e447c4a1355fccb04ee024fa7e7d28c9e4c5ab53347faf",
    "nvperf_host.dll": "cde4143748734948838ede61b350db59a0d7e6ac6ef451e375a5702d85174a9b",
    "nvperf_target.dll": "049f858e512592e895b5789d0872f44ee7cbb57d3e4b8936303f4d7eb5b2b1a7",
}


def parse_smoke(text):
    if len(text) > 4096:
        raise ValueError("pc_preflight_output_limit")
    rows = [json.loads(line, object_pairs_hook=_object) for line in text.splitlines() if line.strip()]
    if len(rows) != 3 or rows[1] != {"stage": "before_empty_kernel"}:
        raise ValueError("pc_preflight_records")
    if set(rows[0]) != {"profiler_initialize", "device_support_result", "device_support_level"} or set(rows[2]) != {
            "launch", "synchronize", "data", "unload", "disable", "destroy"}:
        raise ValueError("pc_preflight_fields")
    values = dict(rows[0], **rows[2])
    if any(type(value) is not int or not -1 <= value <= 2**31-1 for value in values.values()):
        raise ValueError("pc_preflight_values")
    return values


def make_report(capture, text, provenance):
    report = dict(schema_version=1, report_type="xvram.cuda_pc_preflight", version="0.1.0-dev",
                  provenance=provenance, observation=None,
                  capture={key: capture.get(key) for key in ("exit_code", "timed_out", "controller_reaped",
                      "process_tree_drained", "output_truncated", "errors")},
                  proof=dict(memory_bounds=False, cubin_binding=False, device_ordering=False, trace_completeness=False),
                  outcome=dict(status="failed", exit_code=27))
    safe = (type(capture.get("exit_code")) is int and capture.get("timed_out") is False
            and capture.get("controller_reaped") is True and capture.get("process_tree_drained") is True
            and capture.get("output_truncated") is False and capture.get("errors") == [])
    if capture.get("timed_out") is True:
        report["outcome"] = dict(status="timeout", exit_code=26)
        return report
    try:
        values = parse_smoke(text)
        report["observation"] = values
        prerequisites = (values["profiler_initialize"] == values["device_support_result"] == 0
                         and values["device_support_level"] == 3)
        cuda_done = all(values[key] == 0 for key in ("launch", "synchronize", "unload", "destroy"))
        if safe and prerequisites and cuda_done:
            if values["data"] == values["disable"] == 0 and capture["exit_code"] == 0:
                report["outcome"] = dict(status="ready", exit_code=0)
            elif values["data"] == 35 and values["disable"] in (0, 35) and capture["exit_code"] == 27:
                report["outcome"] = dict(status="permission_required", exit_code=23)
    except (ValueError, KeyError, TypeError):
        pass
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        probe = args.probe.resolve()
        if os.name != "nt" or args.probe.is_symlink() or probe.name != "xvram-pc-sampling-probe.exe":
            raise ValueError("pc_preflight_profile")
        provenance = {probe.name: sha256_file(probe)}
        for name, expected in PINS.items():
            path = probe.parent / name
            if path.is_symlink() or sha256_file(path) != expected:
                raise ValueError("pc_preflight_dependency_pin")
            provenance[name] = expected
        args.output_dir.mkdir(parents=True, exist_ok=False)
        capture = run_process([str(probe), "--launch-smoke"], output_dir=args.output_dir,
                              timeout_seconds=30, environment=clean_capture_environment())
        path = args.output_dir / "stdout.txt"
        text = path.read_text(encoding="utf-8") if path.exists() and path.stat().st_size <= 4096 else ""
        report = make_report(capture, text, provenance)
        if sha256_file(probe) != provenance[probe.name]:
            report["outcome"] = dict(status="failed", exit_code=27)
        with (args.output_dir / "pc-preflight.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"PC sampling preflight: {report['outcome']['status']}")
        return report["outcome"]["exit_code"]
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"PC preflight failed: {type(error).__name__}", file=sys.stderr)
        return 74 if isinstance(error, OSError) else 23


if __name__ == "__main__":
    raise SystemExit(main())
