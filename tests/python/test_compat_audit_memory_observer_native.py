"""CPU-only, bounded tests of the real typed CUPTI memory callback observer.

Official headers supply parameter layouts; this helper links no NVIDIA library
and performs no GPU calls. Allocation metadata remains distinct from tensor
bounds, access semantics, and retirement proof.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import jsonschema

from xvram.compat_audit_capture import run_process
from xvram.compat_audit_memory_witness import analyze_rows, error_details


HELPER = os.environ.get('XVRAM_MEMORY_OBSERVER_TEST_EXE')


@unittest.skipUnless(HELPER, 'explicit native typed memory observer helper required')
class NativeMemoryObserverTests(unittest.TestCase):
    def run_observer(self, scenario=None, *, existing=False):
        executable = Path(HELPER).resolve(strict=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / 'memory.jsonl'
            if existing:
                trace.write_bytes(b'existing evidence must remain unchanged\n')
            command = [str(executable)] + ([scenario] if scenario else [])
            capture = run_process(command, output_dir=root / 'capture',
                                  environment={**os.environ, 'XVRAM_MEMORY_WITNESS_TRACE': str(trace)},
                                  timeout_seconds=10, sample_gpu=False)
            self.assertIs(capture['controller_reaped'], True)
            self.assertIs(capture['process_tree_drained'], True)
            self.assertEqual(capture['errors'], [])
            self.assertFalse(capture['timed_out'])
            contents = trace.read_text(encoding='utf-8')
        return capture, contents

    def rows(self, scenario=None):
        capture, contents = self.run_observer(scenario)
        self.assertEqual(capture['exit_code'], 0, contents)
        rows = [json.loads(line) for line in contents.splitlines()]
        schema = json.loads((Path(__file__).resolve().parents[2] / 'schemas' /
                             'cuda-memory-witness-v1.schema.json').read_text(encoding='utf-8'))
        validator = jsonschema.Draft202012Validator(schema)
        for row in rows:
            validator.validate(row)
        self.assertEqual([row['sequence'] for row in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(row['schema_version'] == 1 and
                            row['record_type'] == 'xvram.cuda_memory_witness' for row in rows))
        self.assertEqual(rows[0]['kind'], 'session')
        self.assertEqual(rows[-1]['kind'], 'summary')
        self.assertFalse(rows[0]['host_pinned_observed'])
        self.assertFalse(rows[-1]['terminal_complete'])
        self.assertFalse(rows[-1]['tensor_bounds_proven'])
        # Synthetic raw addresses, context, and handles must never escape as
        # numbers or strings. Normalized process-local object IDs are allowed.
        forbidden = {0x11000000, 0x22000000, 0x33000000, 0x600000, 0xABCDEF55}
        for row in rows:
            self.assertFalse({'address', 'pointer', 'context', 'handle'} & row.keys())
            for value in row.values():
                if type(value) is int:
                    self.assertNotIn(value, forbidden)
        self.assertNotIn('0x', contents)
        return rows

    def test_real_typed_callbacks_and_privacy(self):
        rows = self.rows()
        summary = rows[-1]
        self.assertEqual(summary['api_pairs'], 14)
        self.assertEqual(summary['errors'], 0)
        self.assertEqual(summary['failed_apis'], 0)
        self.assertEqual(summary['open_calls'], 0)
        for name in ('live_allocations', 'live_reservations', 'live_handles', 'live_mappings'):
            self.assertEqual(summary[name], 0)
        self.assertEqual(sum(row['kind'] == 'access_descriptor' for row in rows), 1)
        counts, calls = analyze_rows(rows)
        self.assertEqual(counts['api_pairs'], 14)
        self.assertEqual(len(calls), 14)

    def test_single_native_compound_unmap_retires_three_original_mappings(self):
        rows = self.rows('compound-unmap')
        counts, calls = analyze_rows(rows)
        self.assertEqual((counts['api_pairs'], counts['revision']), (13, 15))
        self.assertEqual((counts['unmap_calls'], counts['unmapped_mappings']), (1, 3))
        self.assertEqual(counts['physical_bytes'], 0)
        self.assertEqual(counts['live_mappings'], 0)
        self.assertEqual(calls[12], ('cuMemUnmap', 0))
        operation = next(row for row in rows if row.get('symbol') == 'cuMemUnmap')
        self.assertEqual(operation['mapping_count'], 3)
        self.assertNotIn('mapping_id', operation)
        segments = [row for row in rows if row['kind'] == 'unmap_segment']
        self.assertEqual([row['index'] for row in segments], [0, 1, 2])
        self.assertEqual([row['offset_bytes'] for row in segments], [0, 4096, 8192])

    def test_failed_api_does_not_admit_output(self):
        rows = self.rows('failed-allocation')
        summary = rows[-1]
        self.assertEqual(summary['errors'], 0)
        self.assertEqual(summary['failed_apis'], 1)
        self.assertEqual(summary['live_allocations'], 0)
        self.assertEqual(analyze_rows(rows)[0]['failed_apis'], 1)

    def test_invalid_state_fails_closed(self):
        for scenario in ('unmapped-copy', 'stale-api', 'unknown-map', 'partial-free', 'stale-free',
                         'unknown-free', 'unsupported'):
            with self.subTest(scenario=scenario):
                rows = self.rows(scenario)
                self.assertGreater(rows[-1]['errors'], 0)
                self.assertTrue(any(row['kind'] == 'error' for row in rows))
                with self.assertRaises(ValueError):
                    analyze_rows(rows)

    def test_free_failure_input_is_disambiguated_without_raw_pointer(self):
        for scenario, null, known, offset in [('unknown-free', False, False, 0),
                                             ('partial-free', False, True, 16)]:
            with self.subTest(scenario=scenario):
                error = next(row for row in self.rows(scenario) if row['kind'] == 'error')
                details = error_details(error)
                self.assertEqual(details['api_id'], 2)
                self.assertEqual(error['category'], 'memory_unknown_or_partial_free')
                self.assertEqual(error['symbol'], 'cuMemFree_v2')
                self.assertEqual(error['callback_stage'], 'exit')
                self.assertTrue(error['parameters_available'])
                self.assertEqual(error['input_kind'], 'address')
                self.assertIs(error['null_input'], null)
                self.assertIs(error['known_identity'], known)
                self.assertEqual(error['offset_bytes'], offset)
                self.assertEqual(error['allocation_id'], 1 if known else 0)
                self.assertEqual(error['generation'], 1 if known else 0)
                self.assertEqual(error['allocation_bytes'], 4096 if known else 0)

    def test_successful_null_free_is_observed_noop_only(self):
        rows = self.rows('null-free')
        counts, calls = analyze_rows(rows)
        self.assertEqual(counts['null_free_noops'], 2)
        self.assertEqual(counts['revision'], 2)
        self.assertEqual(counts['live_allocations'], 0)
        self.assertEqual(counts['failed_apis'], 1)
        self.assertEqual(calls[4], ('cuMemFree_v2', 1))
        noops = [row for row in rows if row.get('no_op') is True]
        self.assertEqual([row['api_id'] for row in noops], [2, 3])
        for row in noops:
            self.assertTrue(row['null_input'])
            self.assertEqual(row['revision'], 1)
            self.assertFalse({'object_id', 'generation', 'bytes'} & row.keys())

    def test_existing_trace_is_never_overwritten(self):
        capture, contents = self.run_observer(existing=True)
        self.assertNotEqual(capture['exit_code'], 0)
        self.assertEqual(contents, 'existing evidence must remain unchanged\n')


if __name__ == '__main__':
    unittest.main()
