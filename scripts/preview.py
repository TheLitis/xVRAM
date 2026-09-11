"""Run the bounded developer preview; missing GPU evidence is never success."""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
MAX_REPORT = 8 * 1024 * 1024
MAX_TRACE = 64 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        require(math.isfinite(result), "non-finite JSON number")
        return result

    def constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    value = json.loads(text, object_pairs_hook=pairs, parse_float=number,
                       parse_constant=constant)
    require(isinstance(value, dict), "expected a JSON object")
    return value


def read_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size <= MAX_REPORT, "missing or oversized JSON file")
    return strict_json(path.read_text(encoding="utf-8"))


def safe_path(root: Path, name: str) -> Path:
    require(isinstance(name, str) and bool(name), "empty package path")
    require(not any(c in name for c in "\\:\x00"), "invalid package path")
    parts = name.split("/")
    require(all(p not in ("", ".", "..") for p in parts), "unsafe package path")
    require(not PurePosixPath(name).is_absolute(), "absolute package path")
    result = root.joinpath(*parts)
    require(result.resolve().is_relative_to(root.resolve()), "package path escapes root")
    require(not any(root.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)),
            "package symlinks are not supported")
    return result


def verify_package(root: Path) -> dict:
    """Inventory integrity, not a digital signature or independent GPU attestation."""
    root = root.resolve()
    manifest = read_json(root / "package-manifest.json")
    require(manifest.get("format") == "xvram.developer_preview.package.v1", "unknown package format")
    require(isinstance(manifest.get("source_commit"), str) and
            re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is not None,
            "full source commit required")
    require(isinstance(manifest.get("sdk_version"), str), "missing SDK version")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "empty manifest")
    require("package-manifest.json" not in files, "self-referential manifest")
    required = {"preview.py", "requirements-preview.txt", "tools/compat_contract.py",
                "share/xvram/schemas/cuda-compat-v1.schema.json",
                "share/xvram/schemas/cuda-compat-trace-v1.schema.json"}
    require(required.issubset(files), "preview tools or schemas missing")
    for name, expected in files.items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "invalid file digest")
        path = safe_path(root, name)
        require(path.is_file() and sha256(path) == expected, f"package integrity failure: {name}")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "unexpected package symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(files) | {"package-manifest.json"}, "unlisted or missing package files")
    return manifest


def load_contract(root: Path):
    # Reuse, do not weaken, the established Phase 6a contract (which uses asserts).
    require(__debug__, "run without -O or PYTHONOPTIMIZE; contract assertions are required")
    spec = importlib.util.spec_from_file_location("xvram_preview_contract", root / "tools/compat_contract.py")
    require(spec is not None and spec.loader is not None, "contract module missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_report(root: Path, manifest: dict, report_path: Path, trace_path: Path,
                    *, exit_code: int | None = None, oversubscribe: bool = False) -> dict:
    report = read_json(report_path)
    contract = load_contract(root)
    schemas = root / "share/xvram/schemas"
    contract.validator(schemas / "cuda-compat-v1.schema.json").validate(report)
    require(report["build"]["git_commit"] == manifest["source_commit"][:12], "report source revision differs")
    require(report["build"]["version"] == manifest["sdk_version"], "report SDK version differs")
    if exit_code is not None:
        require(report["outcome"]["exit_code"] == exit_code, "process/report exit codes disagree")
    contract.validate_semantics(report)
    require(report["outcome"]["exit_code"] == 0, "GPU proof did not complete: " + report["outcome"]["reason"])
    require(report["outcome"]["status"] == "completed", "incomplete GPU report")
    require(report["device"]["total_memory_bytes"] > 0, "no observed device memory")
    require(report["execution"]["telemetry_observed"], "no execution telemetry")
    require(report["execution"]["tiles_retired"] > 0 and bool(report["workloads"]), "empty execution")
    require(report["verification"]["output_elements_checked"] > 0, "no numerical output checked")
    require(not report["configuration"]["identifiers_included"], "preview must redact identifiers")
    require(trace_path.is_file() and trace_path.stat().st_size <= MAX_TRACE, "missing or oversized trace")
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            require(len(line) <= 256 * 1024, "oversized trace record")
            strict_json(line)
    contract.validate_trace(trace_path, contract.validator(schemas / "cuda-compat-trace-v1.schema.json"), report)
    if oversubscribe:
        require(any(w["logical_bytes"] > report["device"]["total_memory_bytes"] and
                    w["evictions"] > 0 and w["handle_reuses"] > 0 for w in report["workloads"]),
                "run did not demonstrate data larger than physical VRAM")
    return report


def build_command(root: Path, mode: str, output: Path, *, device: int = 0,
                  logical_size: str = "auto", timeout: int = 900) -> list[str]:
    require(mode in ("smoke", "oversubscribe"), "unknown run mode")
    require(0 <= device <= 2**31 - 1 and 1 <= timeout <= 3600, "invalid device or timeout")
    require(logical_size == "auto" or re.fullmatch(r"[1-9][0-9]*(?:B|KiB|MiB|GiB)?", logical_size) is not None,
            "logical size must be auto or a positive integer with B/KiB/MiB/GiB")
    bench = root / "bin" / ("xvram-compat-bench.exe" if os.name == "nt" else "xvram-compat-bench")
    libraries = root / ("bin" if os.name == "nt" else "lib")
    core = libraries / ("cublas64_13.dll" if os.name == "nt" else "libcublas.so.13")
    lt = libraries / ("cublasLt64_13.dll" if os.name == "nt" else "libcublasLt.so.13")
    require(bench.is_file() and core.is_file() and lt.is_file(), "packaged benchmark/cuBLAS pair missing")
    command = [str(bench), "--device", str(device), "--policy", "clock", "--passes", "2",
               "--timeout-seconds", str(timeout), "--cublas-library", str(core),
               "--cublas-lt-library", str(lt), "--json", str(output / "report.json"),
               "--trace", str(output / "trace.jsonl")]
    if mode == "smoke":
        command += ["--scenario", "suite", "--m", "37", "--n", "19", "--k", "67",
                    "--padding", "7", "--offset-elements", "17", "--alpha", "1.25", "--beta", "0.5"]
    else:
        command += ["--scenario", "gemm", "--logical-size", logical_size]
    return command


def run_process(command: list[str], output: Path, timeout: int) -> int:
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=os.name != "nt")
        try:
            return process.wait(timeout=timeout + 30)
        except BaseException:
            # Windows controller owns a kill-on-close Job Object. POSIX additionally
            # kills only this wrapper's private process group, never another user's job.
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait()
            raise


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def render_html(summary: dict, report: dict | None) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    passed = summary["status"] == "passed"
    rows = ""
    device = "No successful GPU measurement is claimed."
    if passed and report is not None:
        device = (f'{esc(report["device"]["name"])} · physical VRAM '
                  f'{report["device"]["total_memory_bytes"] / 2**30:.3f} GiB · '
                  f'peak managed residency {report["cache"]["resident_bytes_peak"] / 2**30:.3f} GiB')
        for w in report["workloads"]:
            cells = [w["name"], f'{w["logical_bytes"] / 2**30:.3f} GiB', w["tiles_retired"],
                     w["evictions"], w["pass_timings"]["median_ms"]]
            rows += "<tr>" + "".join("<td>" + esc(c) + "</td>" for c in cells) + "</tr>"
    title = "GPU contract verified" if passed else "GPU proof not completed"
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>xVRAM · local preview report</title><style>
body{{margin:0;background:#10171d;color:#e8edf0;font:17px/1.6 system-ui,sans-serif}}
main{{max-width:1000px;margin:auto;padding:54px 24px}}h1{{font-size:42px;line-height:1.15}}
.status{{border-left:4px solid #a6bcc7;padding:16px 24px;background:#1a2730;margin:30px 0}}
code{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid #50606a}}.scroll{{overflow-x:auto}}
a{{color:#9ee9d3}}footer{{margin-top:36px;color:#b3c1ca}}</style>
<main><header><p>xVRAM / DEVELOPER PREVIEW</p><h1>{title}</h1><p>{device}</p></header>
<section class="status"><strong>{esc(summary["mode"])}</strong><p>{esc(summary.get("message", ""))}</p>
<p>Source <code>{esc(summary["source_commit"])}</code></p></section><div class="scroll"><table>
<thead><tr><th>Workload</th><th>Operand data</th><th>Retired tiles</th><th>Evictions</th><th>Median pass (ms)</th></tr></thead>
<tbody>{rows}</tbody></table></div><p>Pass timing is not full process time. No general speedup is inferred.</p>
<p>Raw records: <a href="report.json">report.json</a> · <a href="trace.jsonl">trace.jsonl</a> · <a href="run.json">run manifest</a></p>
<footer>Explicit synchronous CUDA/cuBLAS integration. RAM is backing storage; VRAM is the bounded execution tier.
Not transparent application support, a production release or independent hardware attestation.
All processing is local. Review paths and hardware/software details before sharing.</footer></main></html>'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "smoke", "oversubscribe", "validate"))
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--logical-size", default="auto")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--require-oversubscription", action="store_true")
    args = parser.parse_args(argv)
    output = summary = report = None
    try:
        root = args.package.resolve()
        manifest = verify_package(root)
        if args.mode == "verify":
            print("Package checksums verified. NOT a GPU test or a digital signature.")
            return 0
        if args.mode == "validate":
            require(args.report is not None and args.trace is not None, "validate needs --report and --trace")
            validate_report(root, manifest, args.report, args.trace, oversubscribe=args.require_oversubscription)
            print("Report and trace contracts verified; not independent hardware attestation.")
            return 0
        require(args.report is None and args.trace is None, "--report/--trace are for validate only")
        load_contract(root)
        output = (args.output or root.parent / ("xvram-results-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])).resolve()
        require(not output.is_relative_to(root), "put output outside the immutable package")
        command = build_command(root, args.mode, output, device=args.device,
                                logical_size=args.logical_size, timeout=args.timeout_seconds)
        output.mkdir(parents=True, exist_ok=False)
        summary = {"format": "xvram.developer_preview.run.v1", "status": "running", "mode": args.mode,
                   "source_commit": manifest["source_commit"], "files": {}, "command": command,
                   "package_manifest_sha256": sha256(root / "package-manifest.json")}
        write_json(output / "run.json", summary)
        print("Running a real GPU workload. Results:", output, flush=True)
        started = time.monotonic()
        code = run_process(command, output, args.timeout_seconds)
        summary.update(process_elapsed_seconds=time.monotonic() - started, process_exit_code=code)
        report = validate_report(root, manifest, output / "report.json", output / "trace.jsonl",
                                 exit_code=code, oversubscribe=args.mode == "oversubscribe")
        summary.update(status="passed", message="Numerical, trace and resource checks passed. Scope: this run only.")
        result = 0
    except KeyboardInterrupt:
        result = 130
        if summary is not None:
            summary.update(status="failed", message="Interrupted; no successful GPU proof claimed.")
    except Exception as error:
        print(f"Preview failed: {type(error).__name__}: {error}", file=sys.stderr)
        result = 26 if isinstance(error, subprocess.TimeoutExpired) else 27
        if summary is not None:
            summary.update(status="failed", message=f"{type(error).__name__}: {error}")
            if summary.get("process_exit_code") in (23, 24, 25, 26, 27, 64, 70, 74):
                result = summary["process_exit_code"]
    if summary is not None and output is not None:
        try:
            for name in ("report.json", "trace.jsonl", "stdout.log", "stderr.log"):
                path = output / name
                if path.is_file():
                    summary["files"][name] = sha256(path)
            summary["runner_exit_code"] = result
            write_json(output / "run.json", summary)
            (output / "report.html").write_text(render_html(summary, report), encoding="utf-8")
            print("Open", output / "report.html")
        except OSError as error:
            print(f"Cannot finalize report: {error}", file=sys.stderr)
            return 74
    return result


if __name__ == "__main__":
    raise SystemExit(main())
