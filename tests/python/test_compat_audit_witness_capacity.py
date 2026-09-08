"""Versioned diagnostic capacity is opt-in; old trace admission stays frozen."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jsonschema

from xvram import compat_audit_identity as identity
from xvram import compat_launch_census as census


ROOT = Path(__file__).resolve().parents[2]


def census_rows(version):
    api = dict(api_id=1, parent_api_id=0, correlation_id=1, probe_call_id=1,
               domain='driver', symbol='cuLaunchKernel')
    rows = [dict(kind='session', terminal_complete=False, cupti_sha256=census.CUPTI_SHA256),
            dict(kind='api_enter', **api), dict(kind='api_exit', result=0, **api),
            dict(kind='kernel', correlation_id=1, name='kernel', start_ns=1, end_ns=2),
            dict(kind='summary', errors=0, dropped=0, buffers_outstanding=0, terminal_complete=False)]
    return [dict(schema_version=version, record_type='xvram.cuda_launch_census', sequence=i, **row)
            for i, row in enumerate(rows, 1)]


class WitnessCapacityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'trace.jsonl'
        self.census_path = self.path.with_name('census.jsonl')

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def test_version_limit_table_is_exact_and_frozen(self):
        self.assertEqual(census.TRACE_LIMITS, {
            1: (128*1024*1024, 750000, 250000),
            2: (256*1024*1024, 1500000, 500000),
            3: (1024*1024*1024, 4000000, 2000000)})

    def test_native_v3_rows_and_old_report_schema(self):
        rows = census_rows(3)
        self.write(self.path, rows)
        observation = census.analyze(self.path, 1)
        self.assertEqual(observation['counts']['api_pairs'], 1)
        schema = json.loads((ROOT / 'schemas/cuda-launch-census-trace-v3.schema.json').read_text())
        validator = jsonschema.Draft202012Validator(schema)
        validator.check_schema(schema)
        for row in rows:
            validator.validate(row)
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(dict(row, native_handle=123))
        for field, value in [('api_id', 2000001), ('parent_api_id', 2000001), ('sequence', 4000001)]:
            with self.assertRaises(jsonschema.ValidationError):
                validator.validate(dict(rows[1], **{field: value}))
        report = dict(schema_version=1, report_type='xvram.cuda_launch_census', version='0.1.0-dev',
                      cupti_sha256=census.CUPTI_SHA256, observation=observation, diagnostics=[],
                      proof=dict.fromkeys(census.PROOF, False), exit_code=0, probe_trace_sha256='a'*64)
        jsonschema.validate(report, json.loads((ROOT / 'schemas/cuda-launch-census-report-v1.schema.json').read_text()))

    def test_byte_limits_preserved_per_version(self):
        for version, (cap, _, _) in census.TRACE_LIMITS.items():
            self.write(self.path, census_rows(version))
            with mock.patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=cap)):
                self.assertEqual(census.analyze(self.path, 1)['counts']['api_pairs'], 1)
            with mock.patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=cap+1)):
                with self.assertRaises(ValueError):
                    census.analyze(self.path, 1)

    def test_record_and_api_limits_are_enforced_without_large_test_allocations(self):
        for version in census.TRACE_LIMITS:
            self.write(self.path, census_rows(version))
            cap, _, _ = census.TRACE_LIMITS[version]
            for limited in ((cap, 4, 1), (cap, 5, 0)):
                with mock.patch.dict(census.TRACE_LIMITS, {version: limited}):
                    with self.assertRaises(ValueError):
                        census.analyze(self.path, 1)
            with mock.patch.dict(census.TRACE_LIMITS, {version: (cap, 5, 1)}):
                self.assertEqual(census.analyze(self.path, 1)['records'], 5)

    def test_versions_cannot_change_mid_trace(self):
        for version in (1, 2, 3):
            rows = census_rows(version)
            rows[-1]['schema_version'] = 2 if version == 3 else 3
            self.write(self.path, rows)
            with self.assertRaisesRegex(ValueError, 'version_changed'):
                census.analyze(self.path, 1)
            with self.assertRaisesRegex(ValueError, 'version_changed'):
                list(identity.records(self.path, 'xvram.cuda_launch_census', versions=(1, 2, 3)))

    def test_generic_limit_is_explicit_and_framing_stays_strict(self):
        self.write(self.path, census_rows(3))
        args = dict(versions=(3,), max_records=5)
        self.assertEqual(len(list(identity.records(self.path, 'xvram.cuda_launch_census', **args))), 5)
        for field, value in [('max_records', 4), ('cap', 1), ('max_records', True), ('cap', 0)]:
            with self.assertRaises(ValueError):
                list(identity.records(self.path, 'xvram.cuda_launch_census', **dict(args, **{field: value})))
        with self.assertRaises(ValueError):
            list(identity.records(self.path, 'xvram.cuda_launch_census'))

    def identity_fixture(self, version, api_id):
        rows = [dict(kind='session', terminal_complete=False),
                dict(kind='library_api', api_id=api_id, symbol='cuLibraryLoadData', result=0,
                     library_id=1, library_generation=1),
                dict(kind='launch_identity', call_id=1, library_id=1, library_generation=1,
                     module_observation_id=2, library_module_equal=True, cubin_binding_proven=False),
                dict(kind='summary', errors=0, terminal_complete=False)]
        rows = [dict(schema_version=version, record_type='xvram.cuda_identity_witness', sequence=i, **row)
                for i, row in enumerate(rows, 1)]
        self.write(self.path, rows)
        self.write(self.census_path, [dict(schema_version=3 if version == 2 else 2,
                   record_type='xvram.cuda_launch_census', sequence=1, kind='api_exit',
                   domain='driver', api_id=api_id, symbol='cuLibraryLoadData', result=0)])
        return rows

    def test_identity_api_capacity_requires_v2(self):
        for version, maximum in ((1, 500000), (2, 2000000)):
            self.identity_fixture(version, maximum)
            self.assertEqual(identity.identity(self.path, self.census_path, 1)['library_loads'], 1)
            self.identity_fixture(version, maximum+1)
            with self.assertRaises(ValueError):
                identity.identity(self.path, self.census_path, 1)
        rows = self.identity_fixture(2, 2000000)
        schema = json.loads((ROOT / 'schemas/cuda-identity-witness-v2.schema.json').read_text())
        for row in rows:
            jsonschema.validate(row, schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(dict(row, native_pointer=123), schema)

    def test_identity_v1_byte_admission_remains_64mib(self):
        for version, cap in ((1, 64*1024*1024), (2, 256*1024*1024)):
            self.identity_fixture(version, 1)
            original = Path.stat
            def mocked(path, *args, **kwargs):
                return SimpleNamespace(st_size=cap+1) if path == self.path else original(path, *args, **kwargs)
            with mock.patch.object(Path, 'stat', mocked):
                with self.assertRaisesRegex(ValueError, 'version_capacity'):
                    identity.identity(self.path, self.census_path, 1)

    def test_preexisting_census_and_identity_contracts_byte_identical(self):
        expected = {
            'cuda-launch-census-trace-v1.schema.json': '6857eacde169778cc1fa3958150c232940e1726ed3c6d2dd07b30a72aa209665',
            'cuda-launch-census-trace-v2.schema.json': '82da749fadfdce78c809f49c9faaa92f0d7a7d3279da56c170d95d75527909b5',
            'cuda-launch-census-report-v1.schema.json': '4f98c9d213b8c50b27bb6d7b74be0593532e699ac1c07dc7d941a79ceeef025b',
            'cuda-identity-witness-v1.schema.json': 'd80aa838f1c15024844c7f164bd03d157a0bb8f7291d05dbf04bbf0a3348de92'}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / 'schemas' / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
