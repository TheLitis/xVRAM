from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from argparse import ArgumentTypeError
from pathlib import Path
from unittest import mock

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from xvram.torch_bench import _normalize_plan, _parser, parse_size, run_controller
from xvram import torch_bench
from xvram.torch_protocol import MessageType, encode_frame
from xvram.torch_report import EXIT_COMPLETED, EXIT_RUNTIME, EXIT_TIMEOUT


HELPER = ROOT / "tests" / "helpers" / "torch_worker_test_helper.py"
SCHEMA = json.loads(
    (ROOT / "schemas" / "pytorch-inference-report-v1.schema.json").read_text(
        encoding="utf-8"
    )
)
SCHEMA_V2 = json.loads(
    (ROOT / "schemas" / "pytorch-inference-report-v2.schema.json").read_text(
        encoding="utf-8"
    )
)


def plan() -> dict[str, object]:
    return {
        "device": 0,
        "model": "operator-smoke",
        "model_ratio": 0.1,
        "layers": 2,
        "batch": 1,
        "sequence": 8,
        "hidden": 64,
        "intermediate": 128,
        "heads": 4,
        "dtype": "float16",
        "policy": "clock",
        "cache_target_bytes": 1024,
        "chunk_size_bytes": 1024,
        "device_headroom_bytes": 0,
        "scratch_cap_bytes": 1024,
        "prefetch_distance": 0,
        "sdpa_backend": "math",
        "seed": "0x1",
    }


def compressed_plan() -> dict[str, object]:
    value = plan()
    value.update(
        {
            "compression": "capacity",
            "compression_codec": "lz4",
            "state_pattern": "structured",
            "host_store_cap_bytes": 4096,
            "host_headroom_bytes": 1024,
            "compression_scratch_cap_bytes": 1024,
            "codec_slots": 2,
            "codec_workers": 2,
        }
    )
    return value


class TorchControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        inherited = os.environ.get("PYTHONPATH")
        local_pythonpath = str(PYTHON_ROOT)
        if inherited:
            local_pythonpath += os.pathsep + inherited
        self._pythonpath = mock.patch.dict(os.environ, {"PYTHONPATH": local_pythonpath})
        self._pythonpath.start()
        self.addCleanup(self._pythonpath.stop)

    def command(self, mode: str) -> list[str]:
        return [sys.executable, str(HELPER), mode]

    def test_cli_plan_keeps_off_on_v1_and_maps_compression_options_to_v2(self) -> None:
        off = _normalize_plan(_parser().parse_args([]))
        self.assertNotIn("compression", off)

        compressed = _normalize_plan(
            _parser().parse_args(
                [
                    "--compression",
                    "capacity",
                    "--compression-codec",
                    "lz4",
                    "--state-pattern",
                    "structured",
                    "--host-store-cap",
                    "12GiB",
                    "--host-headroom",
                    "4GiB",
                    "--compression-scratch-cap",
                    "128MiB",
                    "--codec-slots",
                    "3",
                    "--codec-workers",
                    "4",
                ]
            )
        )
        self.assertEqual(compressed["compression"], "capacity")
        self.assertEqual(compressed["compression_codec"], "lz4")
        self.assertEqual(compressed["state_pattern"], "structured")
        self.assertEqual(compressed["host_store_cap_bytes"], 12 * 1024**3)
        self.assertEqual(compressed["host_headroom_bytes"], 4 * 1024**3)
        self.assertEqual(
            compressed["compression_scratch_cap_bytes"], 128 * 1024**2
        )
        self.assertEqual(compressed["codec_slots"], 3)
        self.assertEqual(compressed["codec_workers"], 4)

    def assert_schema_valid(self, report: dict[str, object]) -> None:
        schema = SCHEMA_V2 if report.get("schema_version") == 2 else SCHEMA
        jsonschema.Draft202012Validator(schema).validate(report)

    def test_success_reaps_worker(self) -> None:
        result = run_controller(plan(), timeout_seconds=5, worker_command=self.command("success"))
        self.assertEqual(result.exit_code, EXIT_COMPLETED)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertTrue(result.report["execution"]["trace_complete"])
        self.assertEqual(result.trace_records[-1]["operation"], "trace_complete")

    def test_compression_success_uses_xvt2_report_and_trace_contracts(self) -> None:
        result = run_controller(
            compressed_plan(),
            timeout_seconds=5,
            worker_command=self.command("success"),
        )
        self.assertEqual(result.exit_code, EXIT_COMPLETED)
        self.assertEqual(result.report["schema_version"], 2)
        self.assertEqual(result.report["compression"]["policy"], "capacity")
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertTrue(result.report["execution"]["trace_complete"])
        self.assertTrue(result.trace_records)
        self.assertTrue(
            all(record["schema_version"] == 2 for record in result.trace_records)
        )
        self.assertTrue(
            all("source_generation" in record for record in result.trace_records)
        )
        self.assert_schema_valid(result.report)

    def test_compression_crash_still_returns_v2_and_reaps_worker(self) -> None:
        result = run_controller(
            compressed_plan(),
            timeout_seconds=5,
            worker_command=self.command("crash"),
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertEqual(result.report["schema_version"], 2)
        self.assertIn("compression", result.report)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertTrue(
            all(record["schema_version"] == 2 for record in result.trace_records)
        )
        self.assert_schema_valid(result.report)

    def test_compression_timeout_is_v2_and_reaps_worker(self) -> None:
        result = run_controller(
            compressed_plan(),
            timeout_seconds=5,
            stall_timeout_seconds=1,
            worker_command=self.command("hang"),
        )
        self.assertEqual(result.exit_code, EXIT_TIMEOUT)
        self.assertEqual(result.report["schema_version"], 2)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertFalse(result.report["execution"]["trace_complete"])
        self.assert_schema_valid(result.report)

    def test_compression_truncated_protocol_is_v2_failure(self) -> None:
        result = run_controller(
            compressed_plan(),
            timeout_seconds=5,
            worker_command=self.command("truncated"),
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertEqual(result.report["schema_version"], 2)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assert_schema_valid(result.report)

    def test_compression_oversized_protocol_is_v2_failure(self) -> None:
        result = run_controller(
            compressed_plan(),
            timeout_seconds=5,
            worker_command=self.command("oversized"),
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertEqual(result.report["schema_version"], 2)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assert_schema_valid(result.report)

    def test_crash_becomes_valid_failure_report(self) -> None:
        result = run_controller(plan(), timeout_seconds=5, worker_command=self.command("crash"))
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertEqual(result.report["report_type"], "xvram.pytorch_inference")
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assert_schema_valid(result.report)

    def test_truncated_protocol_is_failure(self) -> None:
        result = run_controller(plan(), timeout_seconds=5, worker_command=self.command("truncated"))
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])

    def test_oversized_protocol_is_failure(self) -> None:
        result = run_controller(plan(), timeout_seconds=5, worker_command=self.command("oversized"))
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])

    def test_hang_is_killed_and_reaped(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, stall_timeout_seconds=1, worker_command=self.command("hang")
        )
        self.assertEqual(result.exit_code, EXIT_TIMEOUT)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertFalse(result.report["execution"]["trace_complete"])
        self.assertEqual(result.trace_records[-1]["operation"], "trace_incomplete")

    def test_heartbeat_only_worker_still_hits_retirement_watchdog(self) -> None:
        result = run_controller(
            plan(),
            timeout_seconds=5,
            stall_timeout_seconds=1,
            worker_command=self.command("heartbeat-hang"),
        )
        self.assertEqual(result.exit_code, EXIT_TIMEOUT)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertIn("stopped retiring operations", result.report["outcome"]["message"])
        self.assert_schema_valid(result.report)

    def test_malformed_nested_final_is_safe_failure(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=self.command("malformed-final")
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertEqual(result.report["report_type"], "xvram.pytorch_inference")
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertIn("failed validation", result.report["outcome"]["message"])
        self.assert_schema_valid(result.report)

    def test_final_then_hang_is_timeout_not_success(self) -> None:
        result = run_controller(
            plan(),
            timeout_seconds=5,
            stall_timeout_seconds=1,
            worker_command=self.command("final-hang"),
        )
        self.assertEqual(result.exit_code, EXIT_TIMEOUT)
        self.assertFalse(result.report["execution"]["trace_complete"])
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assert_schema_valid(result.report)

    def test_final_exit_grace_is_independent_of_retirement_watchdog(self) -> None:
        # This is a deadline-arithmetic test, not a Python startup benchmark.
        # Live subprocess success/hang/crash/reap coverage remains above. Feed
        # real protocol frames synchronously and advance a controller-local
        # clock only when the modelled child exits, avoiding OS scheduling races.
        spec = importlib.util.spec_from_file_location("torch_worker_fixture", HELPER)
        assert spec is not None and spec.loader is not None
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        worker_report = helper.completed_report(plan())
        trace = {
            "schema_version": 1,
            "record_type": "xvram.pytorch_trace",
            "sequence": 1,
            "monotonic_ns": 0,
            "kind": "lease",
            "region_id": 1,
            "allocation_id": None,
            "operation": "retired",
            "bytes": 128,
            "reason": "test",
        }
        output = b"".join(
            encode_frame(kind, payload)
            for kind, payload in (
                (MessageType.PROGRESS, {"sequence": 1, "operations_retired": 1}),
                (MessageType.TRACE, {"sequence": 2, "records": [trace]}),
                (MessageType.FINAL, {"sequence": 3, "report": worker_report}),
            )
        )
        process = mock.Mock(
            stdin=io.BytesIO(), stdout=io.BytesIO(output), stderr=None, returncode=None
        )
        process.poll.side_effect = lambda: process.returncode
        clock = mock.Mock()
        clock.monotonic.return_value = 0.0
        clock.monotonic_ns.return_value = 0
        exit_delay = 1.0  # Longer than the 0.5 s retirement watchdog.

        def wait_for_exit(*, timeout: float) -> int:
            if timeout < exit_delay:
                raise subprocess.TimeoutExpired("modelled-worker", timeout)
            clock.monotonic.return_value = exit_delay
            process.returncode = EXIT_COMPLETED
            return EXIT_COMPLETED

        process.wait.side_effect = wait_for_exit

        def synchronous_thread(*, target, args, daemon):
            thread = mock.Mock()
            thread.start.side_effect = lambda: target(*args)
            return thread

        with (
            mock.patch.object(torch_bench.subprocess, "Popen", return_value=process),
            mock.patch.object(torch_bench, "_WindowsJob"),
            mock.patch.object(torch_bench.threading, "Thread", side_effect=synchronous_thread),
            mock.patch.object(torch_bench, "time", clock),
        ):
            result = run_controller(
                plan(),
                timeout_seconds=5,
                stall_timeout_seconds=0.5,
                worker_command=["modelled-worker"],
            )
        process.wait.assert_called_once_with(timeout=2.0)
        self.assertEqual(clock.monotonic(), exit_delay)
        self.assertEqual(result.exit_code, EXIT_COMPLETED)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assert_schema_valid(result.report)

    def test_final_exit_code_must_match_normal_worker_exit(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=self.command("exit-mismatch")
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertFalse(result.report["execution"]["trace_complete"])
        self.assertIn("exit code does not match", result.report["outcome"]["message"])
        self.assert_schema_valid(result.report)

    def test_protocol_sequence_must_increase(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=self.command("nonmonotonic-protocol")
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertIn("sequence", result.report["outcome"]["message"])

    def test_trace_sequence_must_increase(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=self.command("nonmonotonic-trace")
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertIn("trace sequence", result.report["outcome"]["message"])

    def test_success_requires_a_reconciled_worker_trace(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=self.command("no-trace")
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertFalse(result.report["execution"]["trace_complete"])
        self.assertIn("without a trace", result.report["outcome"]["message"])

    def test_success_progress_must_cover_completed_regions(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=self.command("progress-mismatch")
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertIn("progress did not reconcile", result.report["outcome"]["message"])

    def test_worker_spawn_failure_is_a_schema_valid_report(self) -> None:
        result = run_controller(
            plan(), timeout_seconds=5, worker_command=[str(ROOT / "missing-worker.exe")]
        )
        self.assertEqual(result.exit_code, EXIT_RUNTIME)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assert_schema_valid(result.report)

    def test_parse_size_never_truncates_to_zero(self) -> None:
        with self.assertRaises(ArgumentTypeError):
            parse_size("0.1b")
        self.assertEqual(parse_size("1b"), 1)


if __name__ == "__main__":
    unittest.main()
