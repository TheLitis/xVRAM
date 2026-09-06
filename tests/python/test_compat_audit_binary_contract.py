"""No-driver structural, semantic and privacy checks for static binary evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import jsonschema

from xvram import compat_audit_binary as binary
from xvram import compat_audit_binary_contract as contract


def _candidate(index, digest, symbol=1):
    return {"module_index": index, "cubin_sha256": digest,
            "symbol_index": symbol, "section_index": symbol, "code_bytes": 32,
            "parameter_bytes": 24, "parameters": [
                {"ordinal": 0, "offset_bytes": 0, "size_bytes": 8},
                {"ordinal": 1, "offset_bytes": 16, "size_bytes": 8}]}


def _completed():
    report = binary._empty_report()
    report["provenance"].update(backend_sha256=contract.BACKEND_SHA256,
                                tool_sha256=contract.TOOL_SHA256,
                                tool_version=contract.TOOL_VERSION,
                                reports=[{"sha256": "a" * 64, "kernel_activities": 7},
                                         {"sha256": "b" * 64, "kernel_activities": 5}])
    report["modules"] = [
        {"module_index": 2, "sha256": "c" * 64, "size_bytes": 512,
         "elf_flags": 0, "symbol_functions": 4, "selected_functions": 2,
         "parameter_dump_sha256": "f" * 64},
        {"module_index": 6, "sha256": "d" * 64, "size_bytes": 512,
         "elf_flags": 0, "symbol_functions": 4, "selected_functions": 1,
         "parameter_dump_sha256": "1" * 64},
        {"module_index": 9, "sha256": "e" * 64, "size_bytes": 512,
         "elf_flags": 0, "symbol_functions": 0, "selected_functions": 0,
         "parameter_dump_sha256": None}]
    report["kernels"] = [
        {"name": "foo", "activities": 3, "status": "static_layout_only",
         "candidates": [_candidate(2, "c" * 64)],
         "runtime_binding_proven": False, "memory_bounds_proven": False},
        {"name": "bar", "activities": 4, "status": "ambiguous",
         "candidates": [_candidate(2, "c" * 64, 2), _candidate(6, "d" * 64)],
         "runtime_binding_proven": False, "memory_bounds_proven": False},
        {"name": "baz", "activities": 5, "status": "missing", "candidates": [],
         "runtime_binding_proven": False, "memory_bounds_proven": False}]
    report["coverage"].update(sm86_cubins=3, candidate_modules=2, observed_types=3,
                              static_layout_types=1, missing_types=1, ambiguous_types=1)
    report["tools"].update(invocations=5, stdout_bytes=4096,
                           all_direct_children_reaped=True, all_pipes_drained=True)
    report["outcome"].update(status="completed", exit_code=0)
    report["cleanup"] = dict.fromkeys(report["cleanup"], True)
    return report


def _put(report, path, value):
    current = report
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _objects(value, path=()):
    if isinstance(value, dict):
        yield path
        for key, item in value.items():
            yield from _objects(item, path + (key,))
    elif isinstance(value, list):
        for key, item in enumerate(value):
            yield from _objects(item, path + (key,))


class BinaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.schema = json.loads((cls.root / "schemas/cuda-compat-audit-binary-evidence-v1.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

    def reject(self, report, code=None, **kwargs):
        with self.assertRaises(ValueError) as raised:
            contract.validate_report(report, **kwargs)
        self.assertRegex(str(raised.exception), r"^binary_report_[a-z_]+$")
        if code:
            self.assertEqual(str(raised.exception), "binary_report_" + code)

    def test_completed_static_layouts_missing_and_ambiguous_remain_no_go(self):
        report = _completed()
        before = copy.deepcopy(report)
        self.assertIsNone(contract.validate_report(report))
        self.validator.validate(report)
        self.assertEqual(report, before)
        self.assertEqual(report["coverage"]["observed_types"], 3)
        self.assertFalse(report["decision"]["execution_ready"])

    def test_fixed_profile_matches_worker_without_hard_coded_observation_counts(self):
        self.assertEqual(binary.BACKEND_SHA256, contract.BACKEND_SHA256)
        self.assertEqual(binary.TOOL_SHA256, contract.TOOL_SHA256)
        self.assertEqual(binary.TOOL_VERSION, contract.TOOL_VERSION)
        self.assertEqual(binary.LIMITATIONS, list(contract.LIMITATIONS))
        empty = binary._empty_report()
        self.assertEqual(empty["schema_version"], contract.SCHEMA_VERSION)
        self.assertEqual(empty["report_type"], contract.REPORT_TYPE)
        report = _completed()
        report["kernels"] = [report["kernels"][-1]]
        report["provenance"]["reports"] = [{"sha256": "a" * 64, "kernel_activities": 5}]
        for module in report["modules"]:
            module.update(selected_functions=0, parameter_dump_sha256=None)
        report["coverage"].update(candidate_modules=0, observed_types=1,
                                  static_layout_types=0, ambiguous_types=0)
        report["tools"]["invocations"] = 3
        contract.validate_report(report)
        self.validator.validate(report)

    def test_empty_failure_rejected_timeout_and_partial_cleanup_are_honest(self):
        for status, code in (("failed", 27), ("rejected", 23), ("timeout", 26)):
            for reaped in (False, True):
                with self.subTest(status=status, reaped=reaped):
                    report = binary._empty_report()
                    report["outcome"].update(status=status, exit_code=code)
                    report["cleanup"]["worker_reaped"] = reaped
                    contract.validate_report(report)
                    self.validator.validate(report)
        report = _completed()
        report["outcome"].update(status="failed", exit_code=27)
        report["cleanup"]["scratch_removed"] = False
        report["diagnostics"] = ["scratch_cleanup_failed"]
        contract.validate_report(report)
        self.validator.validate(report)

    def test_pending_worker_cleanup_requires_explicit_opt_in(self):
        report = _completed()
        report["cleanup"] = dict.fromkeys(report["cleanup"], False)
        self.reject(report, "cleanup_incomplete")
        contract.validate_report(report, allow_pending_cleanup=True)
        with self.assertRaises(jsonschema.ValidationError):
            self.validator.validate(report)
        report["cleanup"]["worker_reaped"] = True
        self.reject(report, "cleanup_incomplete", allow_pending_cleanup=True)
        self.reject(_completed(), "boolean", allow_pending_cleanup=1)

    def test_no_unknown_or_missing_fields_at_any_object_depth(self):
        base = _completed()
        for path in _objects(base):
            for extra in (True, False):
                with self.subTest(path=path, extra=extra):
                    report = copy.deepcopy(base)
                    current = report
                    for part in path:
                        current = current[part]
                    if extra:
                        current["native_handle"] = "0x12345678"
                    else:
                        del current[next(iter(current))]
                    self.reject(report, "fields")
                    self.assertTrue(list(self.validator.iter_errors(report)))

    def test_primitive_types_are_exact_and_errors_do_not_echo_input(self):
        changes = [(("schema_version",), True), (("schema_version",), 1.0),
                   (("cleanup", "worker_reaped"), 1), (("modules", 0, "module_index"), True),
                   (("kernels", 0, "activities"), 1.0), (("kernels", 0, "activities"), -1),
                   (("kernels", 0, "activities"), 1 << 64),
                   (("tools", "stdout_bytes"), float("nan")), (("diagnostics",), "private_path"),
                   (("modules",), ()), (("decision",), ["0x1234"])]
        for path, value in changes:
            with self.subTest(path=path, value=value):
                report = _completed()
                _put(report, path, value)
                self.reject(report)
        self.reject(None, "fields")

    def test_names_codes_and_hashes_cannot_carry_raw_addresses_or_paths(self):
        for name in ("kernel_0x12345678", "0XABCDEF", "bad\nname", "C:\\private\\kernel",
                     "../../kernel", "x" * 1025, "kernel\u200b", "foo\0", "foo\n"):
            with self.subTest(name=name):
                report = _completed()
                report["kernels"][0]["name"] = name
                self.reject(report, "name_privacy")
                self.assertTrue(list(self.validator.iter_errors(report)))
        for code in ("error_0x1234", "C:\\private", "not a code", "Bad_code", "bad__code", "error\n"):
            with self.subTest(code=code):
                report = binary._empty_report()
                report["diagnostics"] = [code]
                self.reject(report, "diagnostic_privacy")
                self.assertTrue(list(self.validator.iter_errors(report)))
        for value in ("0x" + "a" * 62, "A" * 64, "a" * 64 + "\n", "a" * 63):
            report = _completed()
            report["modules"][0]["sha256"] = value
            self.reject(report, "hash")

    def test_false_go_or_runtime_bounds_are_rejected_everywhere(self):
        changes = [(('decision', 'verdict'), 'GO'), (('decision', 'execution_ready'), True),
                   (('decision', 'oversubscription_proof'), True),
                   (('coverage', 'complete_profile_coverage'), True), (('coverage', 'runtime_bound_types'), 1),
                   (('kernels', 0, 'runtime_binding_proven'), True),
                   (('kernels', 0, 'memory_bounds_proven'), True)]
        for path, value in changes:
            with self.subTest(path=path):
                report = _completed()
                _put(report, path, value)
                self.reject(report)
                self.assertTrue(list(self.validator.iter_errors(report)))

    def test_provenance_pins_reports_and_failure_status_pairing(self):
        changes = [(('provenance', 'upstream_commit'), 'a' * 40),
                   (('provenance', 'backend_name'), 'other.dll'),
                   (('provenance', 'backend_sha256'), 'a' * 64),
                   (('provenance', 'tool_name'), 'other'),
                   (('provenance', 'tool_sha256'), 'b' * 64),
                   (('provenance', 'tool_version'), '13.3.74'),
                   (('provenance', 'reports'), []), (('outcome', 'exit_code'), 27),
                   (('outcome', 'status'), 'success'), (('limitations',), list(contract.LIMITATIONS)[1:])]
        for path, value in changes:
            report = _completed()
            _put(report, path, value)
            self.reject(report)
        report = binary._empty_report()
        report['outcome'].update(status='completed', exit_code=0)
        report['cleanup'] = dict.fromkeys(report['cleanup'], True)
        self.reject(report, 'success_evidence')

    def test_unique_source_module_kernel_candidate_and_function_identities(self):
        changes = [(('provenance', 'reports', 1, 'sha256'), 'a' * 64, 'duplicate_source'),
                   (('modules', 1, 'module_index'), 2, 'duplicate_module'),
                   (('kernels', 1, 'name'), 'foo', 'duplicate_kernel'),
                   (('kernels', 1, 'candidates', 1, 'module_index'), 2, 'duplicate_candidate'),
                   (('kernels', 1, 'candidates', 0, 'symbol_index'), 1, 'duplicate_function_identity'),
                   (('kernels', 1, 'candidates', 0, 'section_index'), 1, 'duplicate_function_identity')]
        for path, value, code in changes:
            report = _completed()
            _put(report, path, value)
            self.reject(report, code)

    def test_identical_module_bytes_at_distinct_embedded_indices_are_allowed(self):
        report = _completed()
        report['modules'][1]['sha256'] = 'c' * 64
        report['kernels'][1]['candidates'][1]['cubin_sha256'] = 'c' * 64
        contract.validate_report(report)

    def test_module_references_sizes_dump_hashes_and_selected_functions_reconcile(self):
        changes = [(('kernels', 0, 'candidates', 0, 'module_index'), 999, 'module_reference'),
                   (('kernels', 0, 'candidates', 0, 'cubin_sha256'), 'd' * 64, 'module_reference'),
                   (('kernels', 0, 'candidates', 0, 'code_bytes'), 513, 'integer'),
                   (('modules', 0, 'selected_functions'), 3, 'selection_reconciliation'),
                   (('modules', 0, 'symbol_functions'), 1, 'integer'),
                   (('modules', 0, 'parameter_dump_sha256'), None, 'hash'),
                   (('modules', 2, 'parameter_dump_sha256'), 'a' * 64, 'constant'),
                   (('modules', 0, 'size_bytes'), 63, 'integer'),
                   (('modules', 0, 'elf_flags'), 1 << 32, 'integer')]
        for path, value, code in changes:
            report = _completed()
            _put(report, path, value)
            self.reject(report, code)

    def test_parameter_ordinals_gaps_overlap_overflow_and_terminal_extent(self):
        base = ('kernels', 0, 'candidates', 0)
        changes = [(base + ('parameters', 1, 'ordinal'), 0, 'parameter_ordinals'),
                   (base + ('parameters', 1, 'offset_bytes'), 7, 'parameter_ranges'),
                   (base + ('parameters', 1, 'size_bytes'), 9, 'parameter_ranges'),
                   (base + ('parameters', 1, 'size_bytes'), 7, 'parameter_extent'),
                   (base + ('parameters', 1, 'size_bytes'), 0, 'integer'),
                   (base + ('parameters', 1, 'offset_bytes'), 65536, 'integer'),
                   (base + ('parameter_bytes',), 65537, 'integer'),
                   (base + ('parameters',), [], 'array')]
        for path, value, code in changes:
            report = _completed()
            _put(report, path, value)
            self.reject(report, code)
        report = _completed()
        report['kernels'][0]['candidates'][0]['parameters'][0]['offset_bytes'] = 4
        contract.validate_report(report)  # Both leading and interior padding.
        self.validator.validate(report)

    def test_status_counts_activities_and_uint64_sums_reconcile(self):
        for key in _completed()['coverage']:
            if key == 'complete_profile_coverage':
                continue
            report = _completed()
            report['coverage'][key] += 1
            self.reject(report, 'coverage_reconciliation')
        report = _completed()
        report['kernels'][0]['activities'] += 1
        self.reject(report, 'activity_reconciliation')
        report = _completed()
        report['kernels'][0]['status'] = 'ambiguous'
        self.reject(report, 'constant')
        report = _completed()
        report['provenance']['reports'][0]['kernel_activities'] = contract.MAX_U64
        self.reject(report, 'counter_overflow')
        report = _completed()
        for item in report['provenance']['reports']:
            item['kernel_activities'] = 0
        for item in report['kernels']:
            item['activities'] = 0
        self.reject(report, 'success_evidence')

    def test_tool_reaping_output_accounting_and_cleanup_are_required(self):
        changes = [(('tools', 'invocations'), 4, 'tool_reconciliation'),
                   (('tools', 'stdout_bytes'), 0, 'tool_reconciliation'),
                   (('tools', 'stdout_bytes'), 300 * 1024 * 1024, 'tool_reconciliation'),
                   (('tools', 'all_direct_children_reaped'), False, 'tool_reconciliation'),
                   (('tools', 'all_pipes_drained'), False, 'tool_reconciliation'),
                   (('cleanup', 'worker_reaped'), False, 'cleanup_order'),
                   (('cleanup', 'scratch_removed'), False, 'cleanup_incomplete'),
                   (('diagnostics',), ['unexpected_tool_warning'], 'success_diagnostics')]
        for path, value, code in changes:
            report = _completed()
            _put(report, path, value)
            self.reject(report, code)
        for key, value in (('stdout_bytes', 1), ('all_pipes_drained', True),
                           ('all_direct_children_reaped', True)):
            report = binary._empty_report()
            report['tools'][key] = value
            self.reject(report, 'tool_reconciliation')

    def test_bounded_lists_and_serialized_evidence_size(self):
        report = _completed()
        report['provenance']['reports'] *= 9
        self.reject(report, 'array')
        report = _completed()
        report['modules'] *= 342
        self.reject(report, 'array')
        report = _completed()
        report['kernels'][0]['candidates'][0]['parameters'] *= 65
        self.reject(report, 'array')
        report = binary._empty_report()
        report['diagnostics'] = ['fixed_failure'] * 65
        self.reject(report, 'array')
        with mock.patch.object(contract, 'MAX_EVIDENCE_BYTES', 100):
            self.reject(binary._empty_report(), 'size_limit')

    def test_runtime_validator_loads_without_site_packages(self):
        code = ("import sys; sys.path.insert(0, " + repr(str(self.root / 'python')) + "); "
                "from xvram.compat_audit_binary_contract import validate_report; "
                "from xvram.compat_audit_binary import _empty_report; "
                "validate_report(_empty_report()); "
                "assert 'jsonschema' not in sys.modules")
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                                capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
