"""Windows no-driver tests of the actual native postmortem observer.

The helper links only postmortem.cpp and the normal Windows/C++ runtime. It
neither loads nor calls CUDA. Every child is bounded and reaped by the existing
controller; concurrency cases use native atomic handshakes, not timing sleeps.
"""
import os
from pathlib import Path
import tempfile
import unittest

from xvram.compat_audit_capture import run_process
from xvram.compat_audit_postmortem import FIELDS, FORMAT, Ledger, decode


HELPER = os.environ.get('XVRAM_POSTMORTEM_TEST_EXE')


@unittest.skipUnless(os.name == 'nt' and HELPER, 'explicit Windows native observer helper required')
class NativePostmortemTests(unittest.TestCase):
    def run_observer(self, scenario, *, timeout=10):
        path = Path(HELPER).resolve(strict=True)
        self.assertTrue(path.is_file())
        with tempfile.TemporaryDirectory() as temporary, Ledger() as ledger:
            capture = run_process([str(path), scenario], output_dir=Path(temporary),
                                  environment={**os.environ, **ledger.environment()},
                                  timeout_seconds=timeout, sample_gpu=False)
            # Only read after run_process has terminated/reaped its complete
            # owned child tree, including the native observer helper.
            self.assertIs(capture['controller_reaped'], True)
            self.assertIs(capture['process_tree_drained'], True)
            snapshot = ledger.mapping[:]
        return snapshot, capture, dict(zip(FIELDS, FORMAT.unpack(snapshot)))

    def test_clean_and_sampled_native_roundtrip(self):
        for scenario, sampled in [('clean', False), ('sampled', True)]:
            with self.subTest(scenario=scenario):
                snapshot, capture, row = self.run_observer(scenario)
                result = decode(snapshot, capture, sampled=sampled)
                self.assertEqual(result['entered'], 2)
                self.assertEqual(result['exited'], 2)
                self.assertEqual(row['retained_names'], 39 if sampled else 0)

    def test_late_activity_survives_successful_exit(self):
        for scenario, field in [('late-api-thread', 'late_entries'),
                                ('late-fatal', 'errors'),
                                ('late-activity', 'late_activity'),
                                ('open-api-after-footer', 'apis_open'),
                                ('open-callback-after-footer', 'callbacks_open')]:
            with self.subTest(scenario=scenario):
                snapshot, capture, row = self.run_observer(scenario)
                self.assertEqual(capture['exit_code'], 0)
                self.assertGreater(row[field], 0)
                with self.assertRaises(ValueError):
                    decode(snapshot, capture)

    def test_seal_rejects_live_api_and_callback_deterministically(self):
        for scenario in ('api-open-at-seal', 'callback-open-at-seal'):
            with self.subTest(scenario=scenario):
                snapshot, capture, row = self.run_observer(scenario)
                self.assertEqual(capture['exit_code'], 0)  # Helper saw native seal throw.
                self.assertGreater(row['errors'], 0)
                self.assertEqual(row['callbacks_open'], 0)
                self.assertEqual(row['apis_open'], 0)
                with self.assertRaises(ValueError):
                    decode(snapshot, capture)

    def test_activity_must_retire_before_final_drain(self):
        for scenario in ('activity-outstanding-at-drain', 'activity-callback-open-at-drain'):
            with self.subTest(scenario=scenario):
                snapshot, capture, row = self.run_observer(scenario)
                self.assertEqual(capture['exit_code'], 0)
                self.assertGreater(row['errors'], 0)
                self.assertEqual(row['activity_outstanding'], 0)
                self.assertEqual(row['activity_callbacks_open'], 0)
                with self.assertRaises(ValueError):
                    decode(snapshot, capture)

    def test_counter_underflow_is_sticky_failure(self):
        for scenario, field in [('activity-underflow', 'activity_outstanding'),
                                ('api-underflow', 'apis_open')]:
            with self.subTest(scenario=scenario):
                snapshot, capture, row = self.run_observer(scenario)
                self.assertLess(row[field], 0)
                self.assertGreater(row['errors'], 0)
                with self.assertRaises(ValueError):
                    decode(snapshot, capture)

    def test_early_exit_crash_and_hang_never_certify_cutoff(self):
        for scenario in ('early-exit', 'crash', 'hang'):
            with self.subTest(scenario=scenario):
                snapshot, capture, row = self.run_observer(scenario, timeout=2 if scenario == 'hang' else 10)
                self.assertEqual(row['sealed'], 0)
                if scenario == 'hang':
                    self.assertIs(capture['timed_out'], True)
                if scenario == 'crash':
                    self.assertEqual(capture['exit_code'], 27)
                with self.assertRaises(ValueError):
                    decode(snapshot, capture)


if __name__ == '__main__':
    unittest.main()
