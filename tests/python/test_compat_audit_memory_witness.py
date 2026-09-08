"""No-driver framing, privacy, lifetime and census-join counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jsonschema

from xvram.compat_audit_memory_witness import analyze, analyze_rows


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / 'schemas/cuda-memory-witness-v1.schema.json').read_text(encoding='utf-8'))


def fixture():
    rows = []

    def row(kind, **fields):
        rows.append(dict(schema_version=1, record_type='xvram.cuda_memory_witness',
                         sequence=len(rows) + 1, kind=kind, **fields))

    def op(api, symbol, revision, **fields):
        row('operation', api_id=api, symbol=symbol, result=0, revision=revision, **fields)

    row('session', scope='device_allocations_and_vmm_only', host_pinned_observed=False,
        tensor_bounds_proven=False, terminal_complete=False)
    op(2, 'cuMemAddressReserve', 1, object_id=1, generation=1, bytes=4096)
    op(4, 'cuMemCreate', 2, object_id=2, generation=1, bytes=4096, device=0)
    mapping = dict(allocation_id=1, generation=1, offset_bytes=0, bytes=4096,
                   mapping_id=3, physical_id=2, physical_generation=1, physical_offset_bytes=0)
    op(6, 'cuMemMap', 3, **mapping)
    access = dict(allocation_id=1, generation=1, offset_bytes=0, bytes=4096)
    op(8, 'cuMemSetAccess', 4, descriptor_count=1, **access)
    row('access_descriptor', api_id=8, revision=4, device=0, flags=3, **access)
    op(10, 'cuMemcpyHtoDAsync_v2', 4, allocation_bytes=4096, zero_bytes=False,
       known=True, mapped=True, readable=True, writable=True, **access)
    # Handle release while its mapping exists is legal. Physical bytes remain
    # live until exact full-range unmap; this is not physical-handle aliasing.
    op(12, 'cuMemRelease', 5, object_id=2, generation=1, bytes=4096)
    op(14, 'cuMemUnmap', 6, **mapping)
    op(16, 'cuMemAddressFree', 7, object_id=1, generation=1, bytes=4096)
    row('summary', errors=0, api_pairs=8, failed_apis=0, open_calls=0, revision=7,
        live_allocations=0, live_reservations=0, live_handles=0, live_mappings=0,
        allocation_bytes=0, reservation_bytes=0, physical_bytes=0, mapped_bytes=0,
        tensor_bounds_proven=False, terminal_complete=False)
    return rows


def compound_fixture():
    rows, mappings = [], []
    def row(kind, **fields):
        rows.append(dict(schema_version=1, record_type='xvram.cuda_memory_witness',
                         sequence=len(rows)+1, kind=kind, **fields))
    def op(api, symbol, revision, **fields):
        row('operation', api_id=api, symbol=symbol, result=0, revision=revision, **fields)
    row('session', scope='device_allocations_and_vmm_only', host_pinned_observed=False,
        tensor_bounds_proven=False, terminal_complete=False)
    op(1, 'cuMemAddressReserve', 1, object_id=1, generation=1, bytes=12288)
    for index in range(3):
        physical = 2+2*index
        api = 2+3*index
        op(api, 'cuMemCreate', api, object_id=physical, generation=1, bytes=4096, device=0)
        mapping = dict(allocation_id=1, generation=1, offset_bytes=index*4096, bytes=4096,
                       mapping_id=physical+1, physical_id=physical, physical_generation=1, physical_offset_bytes=0)
        mappings.append(mapping)
        op(api+1, 'cuMemMap', api+1, **mapping)
        op(api+2, 'cuMemRelease', api+2, object_id=physical, generation=1, bytes=4096)
    op(11, 'cuMemUnmap', 13, allocation_id=1, generation=1, offset_bytes=0, bytes=12288, mapping_count=3)
    for index, mapping in enumerate(mappings):
        row('unmap_segment', api_id=11, index=index, revision=13, **mapping)
    op(12, 'cuMemAddressFree', 14, object_id=1, generation=1, bytes=12288)
    row('summary', errors=0, api_pairs=12, failed_apis=0, open_calls=0, revision=14,
        live_allocations=0, live_reservations=0, live_handles=0, live_mappings=0,
        allocation_bytes=0, reservation_bytes=0, physical_bytes=0, mapped_bytes=0,
        tensor_bounds_proven=False, terminal_complete=False)
    return rows


class MemoryWitnessTests(unittest.TestCase):
    def test_compound_unmap_is_one_api_three_mapping_retirements(self):
        rows = compound_fixture()
        validator = jsonschema.Draft202012Validator(SCHEMA)
        for row in rows:
            validator.validate(row)
        counts, calls = analyze_rows(rows)
        self.assertEqual((counts['unmap_calls'], counts['unmapped_mappings']), (1, 3))
        self.assertEqual((counts['api_pairs'], counts['revision']), (12, 14))
        self.assertEqual(calls[11], ('cuMemUnmap', 0))
        self.assertEqual(counts['physical_bytes'], 0)
        old, _ = analyze_rows(fixture())
        self.assertEqual((old['unmap_calls'], old['unmapped_mappings']), (1, 1))

    def test_compound_unmap_requires_exact_ordered_complete_segments(self):
        for index, key, value in ((11, 'mapping_count', 2), (11, 'mapping_count', True),
                                  (11, 'mapping_count', 4097), (11, 'bytes', 12287),
                                  (11, 'offset_bytes', 16), (11, 'generation', 2),
                                  (12, 'api_id', 12), (12, 'revision', 12),
                                  (12, 'physical_generation', 2), (12, 'bytes', 2048),
                                  (13, 'index', 0), (13, 'mapping_id', 3),
                                  (13, 'offset_bytes', 8192), (14, 'allocation_id', 2)):
            rows = compound_fixture(); rows[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                analyze_rows(rows)
        for index in (12, 13, 14):
            rows = compound_fixture(); del rows[index]
            for sequence, row in enumerate(rows, 1):
                row['sequence'] = sequence
            with self.assertRaises(ValueError):
                analyze_rows(rows)
        rows = compound_fixture(); rows[12], rows[13] = rows[13], rows[12]
        for sequence, row in enumerate(rows, 1):
            row['sequence'] = sequence
        with self.assertRaises(ValueError):
            analyze_rows(rows)

    def test_compound_unmap_segment_privacy(self):
        rows = compound_fixture(); rows[12]['pointer'] = 0x22000000
        with self.assertRaises(ValueError):
            analyze_rows(rows)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(rows[12], SCHEMA)

    def test_compound_unmap_does_not_cover_holes_or_other_reservations(self):
        # Remove the middle map but retain its physical handle lifetime events.
        rows = compound_fixture(); del rows[6]
        for sequence, row in enumerate(rows, 1):
            row['sequence'] = sequence
            if 'revision' in row and sequence >= 7:
                row['revision'] -= 1
        with self.assertRaises(ValueError):
            analyze_rows(rows)
        # A range extending beyond the one original reservation is invalid even
        # if some unrelated allocation could happen to be adjacent natively.
        rows = compound_fixture(); rows[11]['bytes'] = 16384
        with self.assertRaises(ValueError):
            analyze_rows(rows)

    def test_schema_and_semantic_roundtrip(self):
        jsonschema.Draft202012Validator.check_schema(SCHEMA)
        validator = jsonschema.Draft202012Validator(SCHEMA)
        for row in fixture():
            validator.validate(row)
        counts, calls = analyze_rows(fixture())
        self.assertEqual(counts['revision'], 7)
        self.assertEqual(counts['live_handles'], 0)
        self.assertEqual(calls[10], ('cuMemcpyHtoDAsync_v2', 0))

    def test_corruption_rejected(self):
        mutations = [
            (0, 'terminal_complete', True), (0, 'host_pinned_observed', True),
            (0, 'schema_version', True), (1, 'sequence', True),
            (1, 'bytes', 2**64), (2, 'object_id', 1), (3, 'generation', 2),
            (3, 'physical_generation', 2), (3, 'physical_offset_bytes', 16),
            (3, 'bytes', 8192), (4, 'descriptor_count', 2), (5, 'flags', 7),
            (5, 'api_id', 9), (6, 'offset_bytes', 4096), (6, 'mapped', False),
            (6, 'allocation_bytes', 8192), (6, 'zero_bytes', True),
            (7, 'generation', 2), (8, 'bytes', 2048), (9, 'bytes', 2048),
            (10, 'live_handles', 1), (10, 'open_calls', 1), (10, 'errors', 1),
            (10, 'revision', 8), (10, 'tensor_bounds_proven', True),
        ]
        for index, key, value in mutations:
            with self.subTest(index=index, key=key, value=value):
                rows = fixture(); rows[index][key] = value
                with self.assertRaises((ValueError, KeyError)):
                    analyze_rows(rows)

    def test_missing_extra_and_after_summary(self):
        for rows in (fixture()[:-1], fixture()[1:], fixture() + [deepcopy(fixture()[-1])], []):
            with self.assertRaises(ValueError):
                analyze_rows(rows)
        rows = fixture(); rows[3]['pointer'] = 0x12345678
        with self.assertRaises(ValueError):
            analyze_rows(rows)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(rows[3], SCHEMA)

    def test_revision_and_api_reuse(self):
        for index, key, value in ((3, 'revision', 2), (6, 'revision', 5), (6, 'api_id', 8)):
            rows = fixture(); rows[index][key] = value
            with self.assertRaises(ValueError):
                analyze_rows(rows)

    def test_injected_error_never_validates(self):
        rows = fixture()
        rows[3] = dict(schema_version=1, record_type='xvram.cuda_memory_witness', sequence=4,
                       kind='error', api_id=6, reason='observer_state_or_unsupported_api')
        jsonschema.validate(rows[3], SCHEMA)
        with self.assertRaisesRegex(ValueError, 'observer_error'):
            analyze_rows(rows)

    def test_null_free_noop_preserves_revision_and_counters(self):
        rows = fixture()
        row = dict(schema_version=1, record_type='xvram.cuda_memory_witness', kind='operation',
                   api_id=13, symbol='cuMemFree_v2', result=0, revision=5, null_input=True, no_op=True)
        rows.insert(8, row)
        for sequence, item in enumerate(rows, 1):
            item['sequence'] = sequence
        rows[-1]['api_pairs'] += 1
        rows[-1]['null_free_noops'] = 1
        jsonschema.validate(row, SCHEMA)
        counts, calls = analyze_rows(rows)
        self.assertEqual(counts['null_free_noops'], 1)
        self.assertEqual(counts['revision'], 7)
        self.assertEqual(calls[13], ('cuMemFree_v2', 0))
        for key, value in [('null_input', False), ('no_op', False), ('revision', 6),
                           ('result', 1), ('symbol', 'cuMemAddressFree'), ('object_id', 1)]:
            with self.subTest(key=key):
                bad = deepcopy(rows); bad[8][key] = value
                with self.assertRaises(ValueError):
                    analyze_rows(bad)
        for value in (0, 2, True):
            bad = deepcopy(rows); bad[-1]['null_free_noops'] = value
            with self.assertRaises(ValueError):
                analyze_rows(bad)

    def test_validated_census_exact_join_and_loss(self):
        rows = fixture()
        census = [dict(kind='api_exit', domain='driver', api_id=r['api_id'],
                       symbol=r['symbol'], result=r['result']) for r in rows if r['kind'] == 'operation']
        checked = dict(sha256='census-hash', counts=dict(open_api_calls=0, buffers_outstanding=0,
                                                       uncorrelated_kernels=0, unmarked_kernels=0))
        def reader(path, *_args, **_kwargs):
            return iter(rows if str(path) == 'memory' else census)
        with patch('xvram.compat_audit_memory_witness.records', side_effect=reader), \
             patch('xvram.compat_audit_memory_witness.digest', side_effect=lambda path, cap: str(path) + '-hash'), \
             patch('xvram.compat_audit_memory_witness.analyze_census', return_value=checked) as validation:
            result = analyze('memory', 'census', 4)
            validation.assert_called_once_with(Path('census'), 4)
            self.assertFalse(result['memory_bounds'])
            self.assertFalse(result['terminal_complete'])
            census[0]['result'] = 2
            with self.assertRaisesRegex(ValueError, 'api_coverage'):
                analyze('memory', 'census', 4)
            census[0]['result'] = 0
            checked['counts']['buffers_outstanding'] = 1
            with self.assertRaisesRegex(ValueError, 'census_incomplete'):
                analyze('memory', 'census', 4)

    def test_file_framing_rejects_duplicate_keys_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.jsonl'
            path.write_text('{"schema_version":1,"schema_version":1}\n', encoding='utf-8')
            # Census is never opened: reject malformed memory records first.
            with patch('xvram.compat_audit_memory_witness.digest', return_value='hash'):
                with self.assertRaises(ValueError):
                    analyze(path, Path(directory) / 'missing', 1)
            path.write_text(json.dumps(fixture()[0]), encoding='utf-8')
            with patch('xvram.compat_audit_memory_witness.digest', return_value='hash'):
                with self.assertRaises(ValueError):
                    analyze(path, Path(directory) / 'missing', 1)


if __name__ == '__main__':
    unittest.main()
