"""No-driver contracts for raw diagnostic rows, not a replacement for proof checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import jsonschema

from xvram.compat_audit_identity import records
from xvram import compat_audit_order as order
from xvram import compat_audit_pc_identity as pc


ROOT = Path(__file__).resolve().parents[2]
U64_MAX = 2**64 - 1


def envelope(record_type, rows):
    return [dict(schema_version=1, record_type=record_type, sequence=i, **row)
            for i, row in enumerate(rows, 1)]


def pc_rows():
    return envelope('xvram.cuda_pc_witness', [
        dict(kind='session', serialized_diagnostic=True, terminal_complete=False,
             sampling_period=11, hardware_buffer_bytes=512*1024*1024,
             configured_buffer_pcs=4096, result_buffer_pcs=4096),
        dict(kind='preflight', context_present=True, errors=0, copies_submitted=0),
        dict(kind='enabled', stall_reasons=38),
        dict(kind='cubin', cubin_cookie=U64_MAX, bytes=4096, sha256='a'*64),
        dict(kind='sample_identity', observed_after_call=1, correlation_id=7,
             cubin_cookie=U64_MAX, function_index=0, name='_Z6kernelv'),
        dict(kind='collection', observed_after_call=1, pc_records=1, dropped_samples=0),
        dict(kind='summary', errors=0, copies_submitted=1, copies_retired=1,
             terminal_complete=False, sampler_cleanup_proven=True),
    ])


def order_rows():
    rows = [dict(kind='session')]
    api = 0

    def pair(symbol, **fields):
        nonlocal api
        api += 1
        common = dict(api_id=api, thread_id=1, context_id=2, stream_id=0,
                      event_id=0, flags=0, stream_kind='none', symbol=symbol,
                      terminal_tool=False, result=0)
        common.update(fields)
        begin = dict(common, kind='api_enter')
        if symbol == 'cuStreamCreate':
            begin.update(stream_id=0, stream_kind='none')
        rows.extend([begin, dict(common, kind='api_exit')])

    pair('cuDevicePrimaryCtxRetain')
    pair('cuStreamCreate', stream_id=3, stream_kind='explicit', flags=1)
    pair('cuLaunchKernel', stream_id=3, stream_kind='explicit')
    pair('cuStreamSynchronize', stream_id=3, stream_kind='explicit')
    pair('cuStreamDestroy_v2', stream_id=3, stream_kind='explicit')
    rows.append(dict(kind='seal', open_calls=0))
    pair('cuCtxSynchronize', terminal_tool=True)
    rows.append(dict(kind='summary', gpu_drained=True, census_drained=True,
                     sealed=True, errors=0, open_calls=0))
    return envelope('xvram.cuda_order_witness', rows)


class ExecutionWitnessSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validators = {}
        for name in ('pc', 'order'):
            schema = json.loads((ROOT / f'schemas/cuda-{name}-witness-v1.schema.json').read_text())
            jsonschema.Draft202012Validator.check_schema(schema)
            cls.validators[name] = jsonschema.Draft202012Validator(schema)

    def assert_invalid(self, name, row):
        with self.assertRaises(jsonschema.ValidationError):
            self.validators[name].validate(row)

    def test_native_shapes_pass_schema_and_semantic_parsers(self):
        for name, rows, parser in [('pc', pc_rows(), pc.analyze_rows),
                                   ('order', order_rows(), order.analyze_rows)]:
            for row in rows:
                self.validators[name].validate(row)
            counts, _ = parser(rows)
            self.assertGreater(counts['collections' if name == 'pc' else 'kernels'], 0)

    def test_all_fields_required_and_cross_variant_fields_forbidden(self):
        for name, rows in [('pc', pc_rows()), ('order', order_rows())]:
            for row in rows:
                for field in row:
                    with self.subTest(name=name, kind=row['kind'], missing=field):
                        altered = dict(row)
                        del altered[field]
                        self.assert_invalid(name, altered)
                # A known field from another variant is still forbidden here.
                field = 'cubin_cookie' if row['kind'] == 'session' else 'serialized_diagnostic'
                if name == 'order':
                    field = 'api_id' if row['kind'] == 'session' else 'sampler_cleanup_proven'
                self.assert_invalid(name, dict(row, **{field: 1}))

    def test_no_raw_addresses_handles_or_extra_success_claims(self):
        for name, rows in [('pc', pc_rows()), ('order', order_rows())]:
            for row in rows:
                for field in ('cuda_address', 'native_pointer', 'stream_handle', 'context_handle',
                              'pCubin', 'proof', 'GO'):
                    with self.subTest(name=name, kind=row['kind'], field=field):
                        self.assert_invalid(name, dict(row, **{field: '0x123456789abc'}))

    def test_boolean_integer_confusion_and_u64_limits(self):
        for name, rows in [('pc', pc_rows()), ('order', order_rows())]:
            for row in rows:
                for field, value in row.items():
                    if type(value) is bool:
                        for invalid in (0, 1, 'true', None):
                            self.assert_invalid(name, dict(row, **{field: invalid}))
                    elif type(value) is int:
                        for invalid in (True, -1 if field != 'result' else -2**31-1,
                                        U64_MAX+1, '1', None):
                            self.assert_invalid(name, dict(row, **{field: invalid}))
        # The transport CRC deliberately uses the whole unsigned 64-bit domain.
        self.validators['pc'].validate(pc_rows()[3])
        self.assert_invalid('pc', dict(pc_rows()[3], cubin_cookie=0))

    def test_capacity_settings_are_bounded(self):
        row = pc_rows()[0]
        for field, values in [('sampling_period', (4, 32)),
                              ('hardware_buffer_bytes', (0, 512*1024*1024+1)),
                              ('configured_buffer_pcs', (0, 4097)),
                              ('result_buffer_pcs', (0, 4097))]:
            for value in values:
                self.assert_invalid('pc', dict(row, **{field: value}))
        for name, rows, maximum in [('pc', pc_rows(), 1500000),
                                     ('order', order_rows(), 4000000)]:
            self.validators[name].validate(dict(rows[0], sequence=maximum))
            self.assert_invalid(name, dict(rows[0], sequence=maximum+1))

    def test_hash_names_and_envelope_are_not_free_form(self):
        for value in ('a'*63, 'A'*64, 'g'*64, 'a'*64+'\n'):
            self.assert_invalid('pc', dict(pc_rows()[3], sha256=value))
        for value in ('', 'x'*4097, 'kernel\naddress'):
            self.assert_invalid('pc', dict(pc_rows()[4], name=value))
        for name, rows in [('pc', pc_rows()), ('order', order_rows())]:
            for field, value in [('schema_version', 2), ('record_type', 'unknown'),
                                  ('kind', 'unknown'), ('sequence', 0)]:
                self.assert_invalid(name, dict(rows[0], **{field: value}))

    def test_failure_records_are_valid_evidence_not_valid_proof(self):
        cases = [('pc', pc_rows(), pc.analyze_rows, [
            dict(kind='error', cupti_result=35),
            dict(kind='copy_rejected', bytes=67108865, active_bytes=0)]),
            ('order', order_rows(), order.analyze_rows, [
                dict(kind='late_producer', api_id=20),
                dict(kind='incomplete', errors=1, open_calls=0)])]
        for name, rows, parser, failures in cases:
            for failure in failures:
                failure.update(schema_version=1, record_type=rows[0]['record_type'],
                               sequence=len(rows))
                self.validators[name].validate(failure)
                with self.assertRaises(ValueError):
                    parser(rows[:-1] + [failure])

    def test_schema_valid_rows_do_not_prove_terminal_or_correlated_state(self):
        rows = pc_rows()
        self.assert_invalid('pc', dict(rows[-1], terminal_complete=True))
        for name, pristine, parser, changes in [
            ('pc', rows, pc.analyze_rows, [(5, 'dropped_samples', 1), (4, 'cubin_cookie', 2),
                                          (-1, 'copies_retired', 0)]),
            ('order', order_rows(), order.analyze_rows,
             [(-1, 'gpu_drained', False), (-1, 'census_drained', False),
              (-1, 'sealed', False), (-1, 'errors', 1)])]:
            for index, field, value in changes:
                altered = copy.deepcopy(pristine)
                altered[index][field] = value
                for row in altered:
                    self.validators[name].validate(row)
                with self.assertRaises(ValueError):
                    parser(altered)

    def test_jsonl_framing_and_terminal_placement(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'trace.jsonl'
            for name, pristine, parser in [('pc', pc_rows(), pc.analyze_rows),
                                           ('order', order_rows(), order.analyze_rows)]:
                raw = ''.join(json.dumps(row)+'\n' for row in pristine)
                for invalid in (raw[:-1], raw.replace('"sequence": 2', '"sequence": 1'),
                                raw.replace('"sequence": 1', '"sequence": 1, "sequence": 1', 1),
                                raw.replace('"sequence": 1', '"sequence": NaN', 1)):
                    path.write_text(invalid, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        list(records(path, pristine[0]['record_type']))
                for rows in (pristine[:-1], pristine + [pristine[-1]], pristine[1:]):
                    with self.assertRaises(ValueError):
                        parser(rows)


if __name__ == '__main__':
    unittest.main()
