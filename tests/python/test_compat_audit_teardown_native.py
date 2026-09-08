"""No-GPU tests of actual typed callback snapshots and true atexit tail."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from xvram.compat_audit_capture import run_process
from xvram.compat_audit_teardown_trace import load_sidecar

HELPER = os.environ.get('XVRAM_TEARDOWN_OBSERVER_TEST_EXE')


@unittest.skipUnless(os.name == 'nt' and HELPER, 'explicit Windows no-driver teardown helper required')
class NativeTeardownTests(unittest.TestCase):
    def run_observer(self, scenario):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / 'teardown.jsonl'
            command = [str(Path(HELPER).resolve(strict=True))]
            if scenario != 'valid':
                command.append(scenario)
            result = run_process(command, output_dir=root / 'capture',
                                 environment={**os.environ, 'XVRAM_TEARDOWN_WITNESS_TRACE': str(trace)},
                                 timeout_seconds=10, sample_gpu=False)
            self.assertIs(result['controller_reaped'], True)
            self.assertIs(result['process_tree_drained'], True)
            self.assertEqual(result['exit_code'], 0)
            self.assertFalse(result['timed_out'])
            self.assertEqual(result['errors'], [])
            data = trace.read_text(encoding='ascii')
            try:
                parsed = load_sidecar(trace)
            except ValueError:
                parsed = None
            return data, parsed

    def test_normal_and_real_atexit_tail_keep_rows_after_checkpoint(self):
        for scenario, count in [('valid', 8), ('atexit-tail', 4)]:
            with self.subTest(scenario=scenario):
                data, parsed = self.run_observer(scenario)
                self.assertIsNotNone(parsed)
                self.assertEqual(len(parsed['typed']), count)
                rows = [json.loads(line) for line in data.splitlines()]
                boundary = next(index for index, row in enumerate(rows) if row['kind'] == 'checkpoint')
                self.assertEqual(len(rows[boundary+1:]), 2)
                self.assertTrue(all(row['kind'] == 'lifecycle' for row in rows[boundary+1:]))
                self.assertNotIn('"terminal_complete":true', data)
                self.assertNotIn('0xABCDEF', data)
                for field in ('native_handle', 'address', 'pid', 'stream_handle'):
                    self.assertTrue(all(field not in row for row in rows))

    def test_generation_and_reference_failures_are_not_valid_observations(self):
        for scenario in ('stale-unload', 'live-library-aba', 'api-aba', 'failed-api', 'orphan-exit',
                         'input-changed', 'unknown-release', 'reference-underflow', 'context-recreation',
                         'wrong-context', 'negative-device', 'multiple-producers'):
            with self.subTest(scenario=scenario):
                data, parsed = self.run_observer(scenario)
                self.assertIsNone(parsed)
                self.assertIn('"kind":"error"', data)

    def test_bounded_host_snapshot_faults_and_open_intervals(self):
        for scenario in ('null-params', 'unreadable-params', 'unreadable-output', 'unreadable-result',
                         'null-output', 'output-slot-changed', 'open-checkpoint', 'overlap'):
            with self.subTest(scenario=scenario):
                _, parsed = self.run_observer(scenario)
                self.assertIsNone(parsed)

    def test_unknown_reset_and_late_queries_or_launches_remain_rejected(self):
        for scenario in ('unknown-reset', 'late-launch', 'late-query'):
            with self.subTest(scenario=scenario):
                data, parsed = self.run_observer(scenario)
                self.assertIsNone(parsed)
                self.assertIn('unsupported_lifecycle_or_tail', data)

    def test_successful_release_with_retained_library_is_observed_not_declared_complete(self):
        data, parsed = self.run_observer('release-before-unloads')
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed['typed']), 3)
        self.assertNotIn('"kind":"error"', data)
        # Native collection does not hide this valid release. The strict epoch
        # model still requires separate evidence for the outstanding library.
        from xvram.compat_audit_teardown import analyze_epoch
        pairs = [dict(api_id=api, parent_api_id=0, domain='driver', symbol=entry[0], result=entry[1],
                      enter_sequence=api*2, exit_sequence=api*2+1)
                 for api, entry in parsed['calls'].items()]
        # Put the execution checkpoint between API2 and API3 without pretending
        # these CPU unit-test observations are a real census/GPU proof.
        pairs[-1].update(enter_sequence=8, exit_sequence=9)
        with self.assertRaises(ValueError):
            analyze_epoch(pairs, parsed['typed'], checkpoint_sequence=6)
