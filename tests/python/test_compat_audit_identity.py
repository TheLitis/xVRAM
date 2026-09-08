import copy
import json
from pathlib import Path
import tempfile
import unittest

import jsonschema
from xvram import compat_audit_identity as identity


class IdentityWitnessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "identity.jsonl"
        self.census = self.root / "census.jsonl"
        self.rows = [dict(kind="session", terminal_complete=False),
                     dict(kind="library_api", api_id=1, symbol="cuLibraryLoadData", result=0,
                          library_id=1, library_generation=1),
                     dict(kind="launch_identity", call_id=1, library_id=1, library_generation=1,
                          module_observation_id=2, library_module_equal=True, cubin_binding_proven=False),
                     dict(kind="summary", errors=0, terminal_complete=False)]
        self.exits = [dict(kind="api_exit", api_id=1, domain="driver", symbol="cuLibraryLoadData", result=0)]
        self.write()

    def write(self):
        for path, kind, version, rows in ((self.path, "xvram.cuda_identity_witness", 1, self.rows),
                                         (self.census, "xvram.cuda_launch_census", 2, self.exits)):
            path.write_text("".join(json.dumps(dict(schema_version=version, record_type=kind, sequence=i, **row))
                                     + "\n" for i, row in enumerate(rows, 1)), encoding="utf-8")

    def analyze(self, calls=1):
        return identity.identity(self.path, self.census, calls)

    def test_actual_library_edge_is_not_a_cubin_proof(self):
        self.assertEqual(self.analyze()["library_linked_launches"], 1)
        probe = dict(capture=dict(timed_out=False), trace=dict(counts=dict(calls_returned=1)), exit_code=0)
        report = identity.make_report(probe, dict(exit_code=0), self.path, self.census)
        self.assertEqual(report["exit_code"], 0)
        self.assertFalse(any(report["proof"].values()))
        schema = Path(__file__).resolve().parents[2] / "schemas/cuda-identity-evidence-v1.schema.json"
        jsonschema.validate(report, json.loads(schema.read_text()))
        report["proof"]["cubin_binding"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(report, json.loads(schema.read_text()))
        trace_schema = schema.parent / "cuda-identity-witness-v1.schema.json"
        for row in identity.records(self.path, "xvram.cuda_identity_witness"):
            jsonschema.validate(row, json.loads(trace_schema.read_text()))
            row["native_pointer"] = 12345
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(row, json.loads(trace_schema.read_text()))

    def test_function_only_is_reported_as_missing_library_edge(self):
        self.rows[2].update(library_id=0, library_generation=0, library_module_equal=False)
        self.write()
        self.assertEqual(self.analyze()["function_only_launches"], 1)

    def test_live_unload_reload_uses_new_generation(self):
        self.rows.insert(3, dict(kind="library_api", api_id=2, symbol="cuLibraryUnload", result=0,
                                 library_id=1, library_generation=1))
        self.rows.insert(4, dict(kind="library_api", api_id=3, symbol="cuLibraryLoadData", result=0,
                                 library_id=1, library_generation=2))
        second = dict(self.rows[2], call_id=2, library_generation=2)
        self.rows.insert(5, second)
        self.exits.extend([dict(self.exits[0], api_id=2, symbol="cuLibraryUnload"), dict(self.exits[0], api_id=3)])
        self.write()
        self.assertEqual(self.analyze(2)["library_linked_launches"], 2)
        second["library_generation"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "unbound_library"):
            self.analyze(2)

    def test_corrupt_fields_ranges_and_false_proofs(self):
        pristine = copy.deepcopy(self.rows)
        for index, key, value in ((0, "terminal_complete", True), (2, "cubin_binding_proven", True),
                                  (2, "library_module_equal", False), (2, "call_id", 2),
                                  (2, "library_generation", 2), (2, "module_observation_id", 0),
                                  (1, "library_generation", True), (1, "api_id", -1),
                                  (1, "symbol", "cuPrivate"), (3, "errors", 1),
                                  (2, "native_pointer", "0x123456789abc")):
            with self.subTest(key=key, value=value):
                self.rows = copy.deepcopy(pristine)
                self.rows[index][key] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.analyze()

    def test_missing_mismatched_or_duplicate_census_api(self):
        for exits in ([], [dict(self.exits[0], api_id=2)], self.exits*2,
                      [dict(self.exits[0], result=1)]):
            self.exits = exits
            self.write()
            with self.assertRaises(ValueError):
                self.analyze()

    def test_failed_library_has_no_success_output(self):
        self.rows[1]["result"] = 2
        self.write()
        with self.assertRaisesRegex(ValueError, "failed_library_output"):
            self.analyze()

    def test_truncated_duplicate_key_nonfinite_and_sequence(self):
        raw = self.path.read_text()
        for data in (raw[:-1], raw.replace('"sequence": 2', '"sequence": 1'),
                     raw.replace('"errors": 0', '"errors": NaN'),
                     raw.replace('"errors": 0', '"errors": 0, "errors": 0')):
            self.path.write_text(data)
            with self.assertRaises(ValueError):
                self.analyze()

    def test_missing_footer_and_late_record_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[2]]):
            self.rows = rows
            self.write()
            with self.assertRaises(ValueError):
                self.analyze()

    def test_timeout_and_missing_sidecar_never_inherit_census_success(self):
        self.path.unlink()
        probe = dict(capture=dict(timed_out=True), trace=None, exit_code=26)
        report = identity.make_report(probe, dict(exit_code=0), self.path, self.census)
        self.assertEqual(report["exit_code"], 26)
        self.assertIsNone(report["observation"])
        self.assertFalse(any(report["proof"].values()))


if __name__ == "__main__":
    unittest.main()
