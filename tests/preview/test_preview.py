"""Synthetic no-GPU fixtures. Never publish these as hardware evidence."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from jsonschema.exceptions import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import preview
import package_preview
REVISION = "1" * 40


def from_schema(schema):
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        if "null" in kind:
            return None
        kind = kind[0]
    if kind == "object":
        return {k: from_schema(schema["properties"][k]) for k in schema.get("required", ())}
    if kind == "array":
        return []
    if kind in ("number", "integer"):
        return schema.get("minimum", 0)
    if kind == "boolean":
        return False
    return "SYNTHETIC_UNIT_TEST_ONLY"


def synthetic_report(oversized=False):
    schema = json.loads((ROOT / "schemas/cuda-compat-v1.schema.json").read_text())
    r = from_schema(schema)
    r["generated_at_utc"] = "2026-09-11T00:00:00Z"
    r["build"].update(git_commit=REVISION[:12], version="0.1.0-dev", build_type="Release")
    r["device"].update(total_memory_bytes=8 if oversized else 16, uuid=None, pci_bus_id=None)
    r["configuration"].update(scenario="gemm", policy="clock", passes=2, trace_enabled=True,
                              cache_target_bytes=8, effective_chunk_bytes=1, staging_slots=2,
                              host_store_cap_bytes=256, host_budget_bytes=64)
    r["proof"] = {k: True for k in r["proof"]}
    r["cleanup"] = {k: True for k in r["cleanup"]}
    r["outcome"].update(status="completed", exit_code=0, reason="synthetic", message="unit test only")
    r["diagnostics"] = []
    r["verification"].update(output_elements_checked=1, max_absolute_error=0, mismatches=0)
    w = from_schema(schema["properties"]["workloads"]["items"])
    w.update(status="completed", m=1, n=1, k=1, logical_bytes=12, storage_bytes=12,
             passes_completed=2, output_elements_checked=1, mismatches=0, tiles_retired=2,
             mappings=1, unmaps=1, h2d_bytes=8, d2h_bytes=4, evictions=1 if oversized else 0,
             handle_reuses=1 if oversized else 0, mismatch_offset=None,
             reference_kind="cpu_fp64_analytic_periodic_v1", digest="a" * 32, reference_digest="a" * 32)
    w["pass_timings"].update(sample_count=2, total_ms=2, minimum_ms=1, median_ms=1, p95_ms=1, maximum_ms=1)
    r["workloads"] = [w]
    r["execution"].update(telemetry_observed=True, gemm_calls=2, tiles_submitted=2, tiles_retired=2,
                          host_backing_peak_bytes=64, trace_complete=True, trace_records=7)
    r["cache"].update(mappings=1, set_access_calls=1, unmaps=1, event_boundaries=1,
                      physical_handles_created=1, physical_handles_released=1,
                      transactions_completed=2, resident_bytes_peak=8, cache_target_bytes=8,
                      cache_target_minimum_bytes=8, cache_target_maximum_bytes=8,
                      bytes_h2d=8, bytes_d2h=4)
    return r


def synthetic_trace():
    rows = []
    for operation in (1, 2):
        for transition in ("call", "retired_tile", "return"):
            rows.append({"schema_version": 1, "report_type": "xvram.cuda_compat_trace",
                         "sequence": len(rows) + 1, "monotonic_timestamp_ns": len(rows),
                         "operation_id": operation, "allocation_id": None, "transition": transition,
                         "bytes": 12, "tiles_retired": operation - 1 if transition == "call" else operation,
                         "reason": "cublasSgemm"})
    rows.append({"schema_version": 1, "report_type": "xvram.cuda_compat_trace", "sequence": 7,
                 "monotonic_timestamp_ns": 7, "operation_id": 0, "allocation_id": None,
                 "transition": "final", "bytes": 0, "tiles_retired": 2, "reason": "synthetic"})
    return "\n".join(json.dumps(row) for row in rows) + "\n"


class PreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.package = self.directory / "package"
        self.package.mkdir()
        for relative, source in {
            "preview.py": "scripts/preview.py", "requirements-preview.txt": "tests/requirements.txt",
            "tools/compat_contract.py": "tests/contract/compat_contract.py",
            "share/xvram/schemas/cuda-compat-v1.schema.json": "schemas/cuda-compat-v1.schema.json",
            "share/xvram/schemas/cuda-compat-trace-v1.schema.json": "schemas/cuda-compat-trace-v1.schema.json",
        }.items():
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, path)
        names = ("bin/xvram-compat-bench.exe", "bin/cublas64_13.dll", "bin/cublasLt64_13.dll") if os.name == "nt" else (
            "bin/xvram-compat-bench", "lib/libcublas.so.13", "lib/libcublasLt.so.13")
        for name in names:
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"SYNTHETIC PLACEHOLDER; NEVER EXECUTE")
        self.manifest = {"format": "xvram.developer_preview.package.v1", "source_commit": REVISION,
                         "sdk_version": "0.1.0-dev", "files": {
                             p.relative_to(self.package).as_posix(): preview.sha256(p)
                             for p in self.package.rglob("*") if p.is_file()}}
        self.write_manifest()
        self.report_path = self.directory / "report.json"
        self.trace_path = self.directory / "trace.jsonl"
        self.save(synthetic_report())
        self.trace_path.write_text(synthetic_trace())

    def write_manifest(self):
        (self.package / "package-manifest.json").write_text(json.dumps(self.manifest))

    def save(self, report):
        self.report_path.write_text(json.dumps(report), encoding="utf-8")

    def validate(self, **kwargs):
        return preview.validate_report(self.package, self.manifest, self.report_path, self.trace_path, **kwargs)

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"n":NaN}',
                     '{"n":Infinity}', '{"n":1e400}', '[]', 'null'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                preview.strict_json(text)
        self.assertEqual(preview.strict_json('{"x":1}'), {"x": 1})

    def test_safe_paths(self):
        for name in ("../x", "/x", "a/../x", "a\\x", "C:x", "a//x", "./x", "a/", "x\0"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                preview.safe_path(self.package, name)

    def test_integrity(self):
        self.assertEqual(preview.verify_package(self.package)["source_commit"], REVISION)
        (self.package / "preview.py").write_text("tampered")
        with self.assertRaises(ValueError):
            preview.verify_package(self.package)

    def test_unlisted_file(self):
        (self.package / "bin/unlisted.dll").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            preview.verify_package(self.package)

    def test_missing_file(self):
        (self.package / "tools/compat_contract.py").unlink()
        with self.assertRaises(ValueError):
            preview.verify_package(self.package)

    def test_missing_critical_tools(self):
        del self.manifest["files"]["preview.py"]
        self.write_manifest()
        with self.assertRaises(ValueError):
            preview.verify_package(self.package)

    def test_self_referential_manifest(self):
        self.manifest["files"]["package-manifest.json"] = "a" * 64
        self.write_manifest()
        with self.assertRaises(ValueError):
            preview.verify_package(self.package)

    @unittest.skipIf(os.name == "nt", "Windows symlink creation requires special policy")
    def test_symlink_escape(self):
        (self.package / "link").symlink_to(self.directory)
        with self.assertRaises(ValueError):
            preview.verify_package(self.package)

    def test_valid_synthetic_contract(self):
        self.assertEqual(self.validate(exit_code=0)["outcome"]["exit_code"], 0)

    def test_oversubscription_not_inferred_from_label(self):
        with self.assertRaises(ValueError):
            self.validate(oversubscribe=True)
        self.save(synthetic_report(oversized=True))
        self.validate(oversubscribe=True)

    def test_wrong_source_and_version(self):
        for key, value in (("git_commit", "unknown"), ("git_commit", "2" * 12), ("version", "1.0.0")):
            r = synthetic_report()
            r["build"][key] = value
            self.save(r)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate()

    def test_process_exit_must_match(self):
        with self.assertRaises(ValueError):
            self.validate(exit_code=27)

    def test_skipped_gpu_not_passed(self):
        r = synthetic_report()
        r["outcome"].update(status="skipped", exit_code=23)
        r["proof"] = {k: None for k in r["proof"]}
        self.save(r)
        with self.assertRaises(ValueError):
            self.validate(exit_code=23)

    def test_contract_corruption(self):
        changes = [("proof", "event_safe", None), ("cleanup", "adapter_closed", False),
                   ("cache", "unsafe_remaps", 1), ("cache", "unmaps", 0),
                   ("execution", "tiles_retired", 3), ("verification", "mismatches", 1),
                   ("execution", "telemetry_observed", False), ("configuration", "identifiers_included", True)]
        for section, key, value in changes:
            r = synthetic_report()
            r[section][key] = value
            self.save(r)
            with self.subTest(section=section, key=key), self.assertRaises((AssertionError, ValueError, ValidationError)):
                self.validate()

    def test_missing_and_invalid_trace(self):
        for text in ("", synthetic_trace() + synthetic_trace(), synthetic_trace().replace('"sequence": 1', '"sequence": 2', 1)):
            self.trace_path.write_text(text)
            with self.subTest(text=text[:20]), self.assertRaises((AssertionError, ValueError, ValidationError)):
                self.validate()
        self.trace_path.unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_trace_key(self):
        self.trace_path.write_text(synthetic_trace().replace('"sequence": 1', '"sequence": 1,"sequence": 1', 1))
        with self.assertRaises(ValueError):
            self.validate()

    def test_optimized_contract_refused(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import preview;from pathlib import Path;preview.load_contract(Path(sys.argv[2]))"
        result = subprocess.run([sys.executable, "-O", "-c", code, str(ROOT / "scripts"), str(self.package)],
                                capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contract assertions are required", result.stderr)

    def test_command_profiles(self):
        smoke = preview.build_command(self.package, "smoke", self.directory)
        self.assertIn("suite", smoke)
        self.assertNotIn("--logical-size", smoke)
        large = preview.build_command(self.package, "oversubscribe", self.directory, logical_size="16GiB")
        self.assertIn("16GiB", large)
        self.assertNotIn("--m", large)
        self.assertNotIn("--include-identifiers", large)
        self.assertNotIn("--test-worker", large)
        for value in ("--help", "0GiB", "1.5GiB", "-1", "1GiB;echo bad"):
            with self.assertRaises(ValueError):
                preview.build_command(self.package, "oversubscribe", self.directory, logical_size=value)

    def test_html_escaping_and_failure(self):
        summary = {"status": "failed", "mode": "<script>alert(1)</script>", "source_commit": REVISION,
                   "message": '<img src=x onerror="bad">'}
        text = preview.render_html(summary, synthetic_report())
        self.assertNotIn("<script>", text)
        self.assertNotIn("<img ", text)
        self.assertIn("No successful GPU measurement", text)
        self.assertNotIn("SYNTHETIC_UNIT_TEST_ONLY", text)
        self.assertIn("Content-Security-Policy", text)

    def test_runner_success_and_revalidation(self):
        output = self.directory / "run"
        def fake_run(command, destination, timeout):
            (destination / "report.json").write_text(json.dumps(synthetic_report()))
            (destination / "trace.jsonl").write_text(synthetic_trace())
            return 0
        with mock.patch.object(preview, "run_process", side_effect=fake_run):
            code = preview.main(["smoke", "--package", str(self.package), "--output", str(output)])
        self.assertEqual(code, 0)
        summary = preview.read_json(output / "run.json")
        self.assertEqual(summary["status"], "passed")
        self.assertIn("trace.jsonl", summary["files"])
        self.assertTrue((output / "report.html").is_file())
        self.assertEqual(preview.main(["validate", "--package", str(self.package), "--report", str(output / "report.json"),
                                      "--trace", str(output / "trace.jsonl")]), 0)
        self.assertEqual(preview.verify_package(self.package)["source_commit"], REVISION)

    def test_runner_cannot_reuse_output(self):
        with mock.patch.object(preview, "run_process") as runner:
            self.assertNotEqual(preview.main(["smoke", "--package", str(self.package), "--output", str(self.directory)]), 0)
            self.assertNotEqual(preview.main(["smoke", "--package", str(self.package), "--output", str(self.package / "results")]), 0)
            runner.assert_not_called()

    def test_failed_process_not_green(self):
        output = self.directory / "failed-run"
        def fake_run(command, destination, timeout):
            r = synthetic_report()
            r["outcome"].update(status="skipped", exit_code=23, reason="synthetic_no_driver")
            r["proof"] = {k: None for k in r["proof"]}
            (destination / "report.json").write_text(json.dumps(r))
            return 23
        with mock.patch.object(preview, "run_process", side_effect=fake_run):
            code = preview.main(["smoke", "--package", str(self.package), "--output", str(output)])
        self.assertEqual(code, 23)
        self.assertEqual(preview.read_json(output / "run.json")["status"], "failed")
        self.assertIn("No successful GPU measurement", (output / "report.html").read_text())

    def test_timeout_and_interrupt(self):
        for error, expected in ((subprocess.TimeoutExpired("synthetic", 1), 26), (KeyboardInterrupt(), 130)):
            output = self.directory / ("error-" + str(expected))
            with mock.patch.object(preview, "run_process", side_effect=error):
                code = preview.main(["smoke", "--package", str(self.package), "--output", str(output)])
            self.assertEqual(code, expected)
            self.assertEqual(preview.read_json(output / "run.json")["status"], "failed")


class PackagingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.install = self.directory / "install"
        for name in package_preview.REQUIRED_INSTALL_FILES:
            path = self.install / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if "schemas/" in name:
                shutil.copy2(ROOT / "schemas" / path.name, path)
            else:
                path.write_text("SYNTHETIC PACKAGE TEST; NOT A BINARY OR LICENSE")

    def test_inventory_archive_and_licenses(self):
        archive = package_preview.assemble(ROOT, self.install, self.directory / "out", REVISION,
                                            "https://github.com/TheLitis/xVRAM/actions/runs/1")
        self.assertTrue(archive.is_file())
        manifest = preview.verify_package(archive.with_suffix(""))
        self.assertEqual(manifest["gpu_validation"], "not_run_by_packaging_ci")
        self.assertIn("LICENSE", manifest["files"])
        self.assertIn("tools/compat_contract.py", manifest["files"])
        self.assertIn(preview.sha256(archive), (archive.parent / "SHA256SUMS.txt").read_text())
        self.assertFalse(any("test_preview.py" in name or "artifacts/" in name for name in manifest["files"]))

    def test_incomplete_install_rejected(self):
        (self.install / "bin/cublasLt64_13.dll").unlink()
        with self.assertRaises(ValueError):
            package_preview.assemble(ROOT, self.install, self.directory / "out", REVISION, "unit-test")

    def test_existing_destination_rejected(self):
        with self.assertRaises(ValueError):
            package_preview.assemble(ROOT, self.install, self.directory, REVISION, "unit-test")

    def test_private_artifact_rejected(self):
        (self.install / "bin/private.pdb").write_bytes(b"private")
        with self.assertRaises(ValueError):
            package_preview.assemble(ROOT, self.install, self.directory / "out", REVISION, "unit-test")


if __name__ == "__main__":
    unittest.main()
