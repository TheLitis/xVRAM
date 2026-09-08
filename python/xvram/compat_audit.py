"""Pinned, observation-only audit of an unchanged llama.cpp CUDA application."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence

from .compat_audit_capture import run_process, sanitize_log

VERSION = "0.1.0-dev"
DEFAULT_PROMPT = (
    "Explain how a computer can process a large collection of data in small pieces. "
    "Discuss main memory, temporary storage, ordering, correctness and reproducibility. "
    "Use a short practical example, then explain why measuring elapsed time separately "
    "from initialization is important. Do not use code. "
) * 3


class AuditPrerequisite(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: {message}\n")


def load_profile() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("compat_audit_profile.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_binary_directory(directory: Path, profile: dict[str, Any]) -> list[Path]:
    expected = profile["upstream"]["files"]
    actual = {item.name for item in directory.iterdir() if item.suffix.lower() in (".exe", ".dll")}
    if actual != set(expected):
        raise AuditPrerequisite("binary directory is not the exact pinned package (missing or extra EXE/DLL)")
    result = []
    for name, digest in sorted(expected.items()):
        item = directory / name
        if item.is_symlink() or not item.is_file() or sha256_file(item) != digest:
            raise AuditPrerequisite(f"binary hash mismatch: {name}")
        result.append(item)
    return result


def verify_model(directory: Path, model: dict[str, Any]) -> list[Path]:
    expected = {item["name"] for item in model["files"]}
    if {item.name for item in directory.glob("*.gguf")} != expected:
        raise AuditPrerequisite("model directory must contain exactly the pinned Q4_K_M shards")
    result = []
    for item in model["files"]:
        path = directory / item["name"]
        if path.is_symlink() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise AuditPrerequisite(f"model hash mismatch: {item['name']}")
        result.append(path)
    return result


def build_command(args: argparse.Namespace, first_shard: Path) -> list[str]:
    executable = args.binary_dir / f"llama-{args.application}.exe"
    command = [str(executable.resolve()), "--model", str(first_shard.resolve()),
               "--ctx-size", str(args.context_size), "--batch-size", "128",
               "--ubatch-size", str(args.microbatch), "--gpu-layers", str(args.gpu_layers),
               "--device", "CUDA0", "--fit", "off", "--flash-attn", "off",
               "--cache-type-k", "f16", "--cache-type-v", "f16",
               "--no-context-shift", "--single-turn", "--simple-io",
               "--no-display-prompt", "--predict", str(args.generate), "--seed", "424242",
               "--temp", "0", "--log-colors", "off", "--color", "off"]
    if args.application == "completion":
        command.append("--no-conversation")
    else:
        command.extend(["--spec-type", "none"])
    if args.prompt_file:
        command.extend(["--file", str(args.prompt_file.resolve())])
    else:
        command.extend(["--prompt", DEFAULT_PROMPT])
    return command


def clean_capture_environment() -> dict[str, str]:
    # A parent shell must not silently override the audited profile or load a
    # different backend/profiler into the native application.
    blocked = ("LLAMA_", "GGML_", "CUDA_INJECTION", "NVTX_INJECTION", "XVRAM_", "NSYS_", "CUPTI_")
    return {name: value for name, value in os.environ.items()
            if not name.startswith(blocked) and name not in ("CUDA_VISIBLE_DEVICES", "CUDA_MANAGED_FORCE_DEVICE_ALLOC")}


def _gpu_sample() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False}
    try:
        result = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                                 "--format=csv,noheader,nounits", "--id=0"], capture_output=True,
                                text=True, timeout=5, check=True)
        name, total, used, free, driver = [part.strip() for part in result.stdout.strip().split(",")]
        return {"available": True, "name": name, "total_bytes": int(total) * 1024**2,
                "used_bytes": int(used) * 1024**2, "free_bytes": int(free) * 1024**2,
                "driver_version": driver, "scope": "whole_device_not_process"}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"available": False}


def capture_platform_supported() -> bool:
    return os.name == "nt" and platform.machine().lower() in ("amd64", "x86_64") and sys.maxsize > 2**32


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None if compact else 2)
    path.write_text(data + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="xvram-compat-audit", description=__doc__)
    parser.add_argument("--stage", choices=("inventory", "capture", "analyze", "all"), default="inventory")
    parser.add_argument("--model", choices=("14b", "32b"), default="14b")
    parser.add_argument("--application", choices=("completion", "cli"), default="completion")
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/compat-audit"))
    parser.add_argument("--collector", type=Path)
    parser.add_argument("--capture-mode", choices=("baseline", "cupti", "nsys"), default="cupti")
    parser.add_argument("--trace-version", type=int, choices=(1, 2, 3), default=1,
                        help="CUPTI trace v2 additionally observes documented dynamic API resolution")
    parser.add_argument("--nsys", type=Path)
    parser.add_argument("--input-trace", type=Path, action="append", default=[])
    parser.add_argument("--microbatch", type=int, choices=(1, 128), default=128)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int, default=8)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--json", default=None)
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 128 <= args.context_size <= 2048 or not 1 <= args.generate <= 256:
        parser.error("context must be 128..2048 and generated tokens 1..256")
    if not 0 <= args.gpu_layers <= 64 or not 0 < args.timeout_seconds <= 86400:
        parser.error("gpu layers must be 0..64 and timeout 0..86400 seconds")
    if args.stage in ("capture", "all") and (args.binary_dir is None or args.model_dir is None):
        parser.error("capture requires --binary-dir and --model-dir")
    from .compat_audit_analysis import AuditInputError, analyze, combine_gguf_metadata, inventory, read_gguf_metadata
    pinned = load_profile()
    args.output_dir = args.output_dir.resolve()
    try:
        if args.stage in ("capture", "all") and args.output_dir.exists() and any(args.output_dir.iterdir()):
            print("capture requires a fresh empty --output-dir", file=sys.stderr)
            return 23
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("cannot create audit output directory", file=sys.stderr)
        return 74
    inventory_data = []
    metadata = {}
    traces = list(args.input_trace)
    result: dict[str, Any] | None = None
    problem: str | None = None
    exit_code = 0
    profile = {"profile_id": pinned["profile_id"], "cache_budget_bytes": None,
               "scratch_reserve_bytes": None, "source_evidence": pinned.get("source_evidence", [])}
    try:
        if args.binary_dir:
            inventory_data = inventory(verify_binary_directory(args.binary_dir, pinned))
        if args.model_dir:
            shards = verify_model(args.model_dir, pinned["models"][args.model])
            metadata = combine_gguf_metadata([read_gguf_metadata(path) for path in shards])
        if args.stage in ("capture", "all"):
            if not capture_platform_supported():
                raise AuditPrerequisite("the pinned hardware capture profile requires Windows x64")
            trace_path = args.output_dir / "cupti-trace.jsonl"
            if trace_path.exists() or (args.output_dir / "capture.json").exists():
                raise AuditPrerequisite("capture output already exists; choose a fresh --output-dir")
            command = build_command(args, shards[0])
            prompt_hash = (sha256_file(args.prompt_file) if args.prompt_file
                           else hashlib.sha256(DEFAULT_PROMPT.encode("utf-8")).hexdigest())
            env = clean_capture_environment()
            env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
            env["PATH"] = str(args.binary_dir.resolve()) + os.pathsep + env.get("PATH", "")
            if args.capture_mode == "cupti":
                if not args.collector or not args.collector.is_file():
                    raise AuditPrerequisite("CUPTI capture requires the built --collector DLL")
                env["CUDA_INJECTION64_PATH"] = str(args.collector.resolve())
                env["XVRAM_AUDIT_TRACE"] = str(trace_path)
                env["XVRAM_AUDIT_TRACE_VERSION"] = str(args.trace_version)
                cupti_dir = Path(os.environ.get("CUDA_PATH", "")) / "extras" / "CUPTI" / "lib64"
                env["PATH"] = str(args.collector.resolve().parent) + os.pathsep + str(cupti_dir) + os.pathsep + env["PATH"]
                traces.append(trace_path)
            elif args.capture_mode == "nsys":
                if not args.nsys or not args.nsys.is_file():
                    raise AuditPrerequisite("Nsight capture requires --nsys executable")
                command = [str(args.nsys.resolve()), "profile", "--trace=cuda,cublas", "--sample=none",
                           "--cpuctxsw=none", "--cuda-memory-usage=true", "--cuda-trace-all-apis=true", "--export=sqlite",
                           "--output=" + str(args.output_dir / "native-nsight")] + command
            before = _gpu_sample()
            result = run_process(command, output_dir=args.output_dir, timeout_seconds=args.timeout_seconds,
                                 environment=env, cwd=args.binary_dir, sample_gpu=True)
            # In this exact pinned completion frontend, --no-display-prompt and
            # --no-conversation leave stdout to generated pieces (the pinned
            # completion.cpp output path). This is not GPU token-completion time.
            if (args.application == "completion" and args.capture_mode == "baseline"
                    and result.get("exit_code") == 0 and not result.get("output_truncated")
                    and (result.get("timings", {}).get("decode_tokens") or 0) > 0
                    and result.get("first_stdout_ms") is not None):
                result["ttft_ms"] = result["first_stdout_ms"]
                result["ttft_status"] = "process_start_to_first_visible_generated_output"
            after = _gpu_sample()
            # Sampling cannot prove every dynamic load, but a seen known module
            # with different bytes invalidates this run's pinned provenance.
            pinned_modules = {name.casefold(): value for name, value in pinned["upstream"]["files"].items()}
            for module in result.get("loaded_modules", []):
                expected = pinned_modules.get(module["name"].casefold())
                if expected is not None and expected != module["sha256"]:
                    result.setdefault("errors", []).append("loaded_module_hash_mismatch")
            if args.capture_mode == "nsys":
                from .compat_audit_nsys import summarize_nsys
                summary = summarize_nsys(args.output_dir / "native-nsight.sqlite")
                _write_json(args.output_dir / "nsight-summary.json", summary)
            observation = {"schema_version": 1, "report_type": "xvram.cuda_compat_audit_capture",
                           "profile_id": pinned["profile_id"], "model": args.model,
                           "application": args.application, "capture_mode": args.capture_mode,
                           "native_cpu_offload": args.gpu_layers <= (48 if args.model == "14b" else 64),
                           "oversubscription_proof": False, "configuration": {
                               "microbatch": args.microbatch, "context_size": args.context_size,
                               "gpu_layers": args.gpu_layers, "generation_limit": args.generate,
                               "seed": 424242, "prompt_sha256": prompt_hash,
                               "graphs": False, "flash_attention": False, "kv_dtype": "f16",
                               "speculative_decoding": False, "context_shift": False,
                           }, "tools": {"controller_version": VERSION,
                                         "collector_trace_version": args.trace_version if args.capture_mode == "cupti" else None,
                                         "python_version": platform.python_version(),
                                         "profile_manifest_sha256": sha256_file(Path(__file__).with_name("compat_audit_profile.json")),
                                         "collector_sha256": sha256_file(args.collector) if args.capture_mode == "cupti" else None,
                                         "nsys_sha256": sha256_file(args.nsys) if args.capture_mode == "nsys" else None},
                           "nsight_summary_sha256": sha256_file(args.output_dir / "nsight-summary.json") if args.capture_mode == "nsys" else None,
                           "direct_command_kind": "profiler" if args.capture_mode == "nsys" else f"llama_{args.application}",
                           "native_exit_code": None if args.capture_mode == "nsys" else result.get("exit_code"),
                           "gpu_before": before, "gpu_after": after, "run": result}
            _write_json(args.output_dir / "capture.json", observation)
            if result.get("timed_out"):
                exit_code = 26
            elif result.get("exit_code") != 0 or result.get("errors") or result.get("output_truncated"):
                exit_code = 27
    except (OSError, ValueError) as error:
        problem = sanitize_log(str(error))
        exit_code = 23 if isinstance(error, (AuditPrerequisite, FileNotFoundError)) else 64
    cleanup = {"controller_reaped": result.get("controller_reaped") if result else None,
               "collector_finalized": None, "complete": None}
    provenance = {"upstream_commit": pinned["upstream"]["commit"],
                  "model_revision": pinned["models"][args.model]["revision"],
                  "profile_id": pinned["profile_id"], "analyzer_version": VERSION,
                  "tool_versions": {"python": platform.python_version(), "xvram_audit": VERSION}}
    observation_file = args.output_dir / "capture.json"
    if observation_file.is_file():
        provenance.update(observation_file=observation_file.name,
                          observation_file_sha256=sha256_file(observation_file))
    try:
        report = analyze(traces, inventory_data, metadata or {}, profile, provenance=provenance, cleanup=cleanup)
    except AuditInputError as error:
        problem = sanitize_log(str(error))
        exit_code = exit_code or 64
        report = analyze([], [], {}, profile, provenance=provenance, cleanup=cleanup)
    # The observer sidecar preserves timing/placement separately from execution
    # readiness. A successful native CPU-offload run cannot turn audit NO-GO to GO.
    if problem:
        report["unresolved"].append({"code": "audit_preflight_failure", "count": 1, "evidence": "unresolved"})
        print(problem, file=sys.stderr)
    if exit_code:
        report["outcome"] = {"status": "failed", "exit_code": exit_code}
        report["decision"]["verdict"] = "NO-GO"
        report["decision"]["execution_ready"] = False
    try:
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=None if args.compact_json else 2)
        if args.json == "-":
            print(encoded)
        else:
            _write_json(Path(args.json) if args.json else args.output_dir / "report.json", report,
                        compact=args.compact_json)
        if not args.no_text:
            print(f"xVRAM compatibility audit: {report['decision']['verdict']} (not an oversubscription run)",
                  file=sys.stderr if args.json == "-" else sys.stdout)
        return exit_code or int(report["outcome"].get("exit_code", 0))
    except (OSError, ValueError):
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
