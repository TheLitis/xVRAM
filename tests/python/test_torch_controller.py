from __future__ import annotations

import json
import subprocess
import sys
import unittest
from argparse import ArgumentTypeError
from pathlib import Path

import jsonschema

from xvram.torch_bench import parse_size, run_controller
from xvram.torch_report import EXIT_COMPLETED, EXIT_RUNTIME, EXIT_TIMEOUT


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tests" / "helpers" / "torch_worker_test_helper.py"
SCHEMA = json.loads(
    (ROOT / "schemas" / "pytorch-inference-report-v1.schema.json").read_text(
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


class TorchControllerTests(unittest.TestCase):
    def command(self, mode: str) -> list[str]:
        return [sys.executable, str(HELPER), mode]

    def assert_schema_valid(self, report: dict[str, object]) -> None:
        jsonschema.Draft202012Validator(SCHEMA).validate(report)

    def test_success_reaps_worker(self) -> None:
        result = run_controller(plan(), timeout_seconds=5, worker_command=self.command("success"))
        self.assertEqual(result.exit_code, EXIT_COMPLETED)
        self.assertTrue(result.report["cleanup"]["worker_reaped"])
        self.assertTrue(result.report["execution"]["trace_complete"])
        self.assertEqual(result.trace_records[-1]["operation"], "trace_complete")

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
        result = run_controller(
            plan(),
            timeout_seconds=5,
            stall_timeout_seconds=0.5,
            worker_command=self.command("final-slow-exit"),
        )
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
