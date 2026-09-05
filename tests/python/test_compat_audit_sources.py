"""No-driver tests for evidence indexing; fixtures prove no real CUDA ABI."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from xvram import compat_audit_sources as sources


class SourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_text = "static __global__ void cpy_scalar(const char * src, char * dst) {}\n"
        self.source_path = "ggml/src/ggml-cuda/cpy.cu"
        path = self.root / self.source_path
        path.parent.mkdir(parents=True)
        path.write_text(self.source_text, encoding="utf-8", newline="\n")
        self.entry = {"path": self.source_path, "sha256": hashlib.sha256(self.source_text.encode()).hexdigest(), "size_bytes": len(self.source_text)}
        self.profile = {"files": [self.entry]}
        self.report = {"report_type": "xvram.cuda_compat_audit", "schema_version": 1,
                       "provenance": {"upstream_commit": sources.COMMIT},
                       "coverage": {"kernel_activities": 3, "activity_kernel_types": [{"name": "_Z10cpy_scalarPKcPc", "activities": 3}]}}
        self.report_path = self.write_report(self.report)

    def write_report(self, value, name="report.json"):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def build(self, reports=None):
        with mock.patch.object(sources, "load_profile", return_value=(self.profile, "a" * 64)):
            return sources.build_index(self.root, reports or [self.report_path])

    def test_packaged_manifest_is_pinned_and_bounded(self):
        profile, digest = sources.load_profile()
        self.assertEqual(profile["upstream_commit"], sources.COMMIT)
        self.assertEqual(len(profile["files"]), 18)
        self.assertEqual(len(digest), 64)
        self.assertLess(sum(item["size_bytes"] for item in profile["files"]), 2 * 1024 * 1024)

    def test_source_profile_rejects_traversal_duplicates_and_unpinned_urls(self):
        profile, _ = sources.load_profile()
        cases = []
        for path in ("../cpy.cu", "ggml/src/../../cpy.cu", "ggml/src//cpy.cu", "ggml/src/cpy.cu\\other"):
            modified = copy.deepcopy(profile)
            modified["files"][0]["path"] = path
            cases.append(modified)
        modified = copy.deepcopy(profile)
        modified["files"].append(modified["files"][0])
        cases.append(modified)
        modified = copy.deepcopy(profile)
        modified["source_base_url"] = "https://example.invalid/"
        cases.append(modified)
        for value in cases:
            with self.subTest(value=value), mock.patch.object(sources, "_bounded_bytes", return_value=json.dumps(value).encode()):
                with self.assertRaises(sources.SourceIndexError):
                    sources.load_profile()

    def test_candidate_is_never_binary_or_memory_proof(self):
        result = self.build()
        kernel = result["kernels"][0]
        self.assertEqual(kernel["status"], "candidate_only")
        self.assertFalse(kernel["binary_binding_proven"])
        self.assertFalse(kernel["memory_bounds_proven"])
        self.assertEqual(kernel["candidate_definitions"][0]["line"], 1)
        self.assertEqual(result["decision"]["verdict"], "NO-GO")
        self.assertFalse(result["decision"]["interception_ready"])
        self.assertFalse(result["coverage"]["complete_profile_coverage"])

    def test_input_go_and_extra_binding_claims_cannot_promote(self):
        self.report["decision"] = {"verdict": "GO", "interception_ready": True}
        self.report["coverage"]["activity_kernel_types"][0]["argument_abi"] = "source_proven"
        self.write_report(self.report)
        self.assertEqual(self.build()["decision"]["verdict"], "NO-GO")

    def test_missing_source_stays_unresolved(self):
        (self.root / self.source_path).unlink()
        result = self.build()
        self.assertEqual(result["sources"][0]["status"], "unreadable")
        self.assertEqual(result["kernels"][0]["status"], "unresolved")
        self.assertTrue(all(item["claim"] is None for item in result["source_facts"]))

    def test_tampered_source_cannot_supply_candidate(self):
        (self.root / self.source_path).write_text(self.source_text + "// altered", encoding="utf-8")
        result = self.build()
        self.assertEqual(result["sources"][0]["status"], "hash_mismatch")
        self.assertEqual(result["kernels"][0]["candidate_definitions"], [])

    def test_ambiguous_definition_never_silently_selects_one(self):
        source = self.source_text * 2
        (self.root / self.source_path).write_text(source, encoding="utf-8", newline="\n")
        self.entry.update(sha256=hashlib.sha256(source.encode()).hexdigest(), size_bytes=len(source))
        kernel = self.build()["kernels"][0]
        self.assertEqual(kernel["status"], "ambiguous_source_candidates")
        self.assertEqual(len(kernel["candidate_definitions"]), 2)
        self.assertFalse(kernel["binary_binding_proven"])

    def test_candidate_identifier_is_not_template_or_nested_abi_parser(self):
        self.assertEqual(sources.candidate_identifier("_Z13mul_mat_vec_qIL9ggml_type12E"), "mul_mat_vec_q")
        self.assertIsNone(sources.candidate_identifier("_ZN7cutlass7Kernel2"))
        self.assertIsNone(sources.candidate_identifier("ampere_h16816gemm"))
        self.assertIsNone(sources.candidate_identifier("_Z999short"))

    def test_opaque_library_kernel_remains_unresolved(self):
        self.report["coverage"]["activity_kernel_types"][0]["name"] = "ampere_h16816gemm_128x64"
        self.write_report(self.report)
        kernel = self.build()["kernels"][0]
        self.assertEqual(kernel["status"], "unresolved")
        self.assertIsNone(kernel["candidate_identifier"])

    def test_activity_counts_not_callback_launch_counts(self):
        self.report["coverage"]["kernel_launches"] = 9000
        self.write_report(self.report)
        self.assertEqual(self.build()["coverage"]["kernel_activities"], 3)

    def test_inconsistent_activity_count_rejected(self):
        self.report["coverage"]["kernel_activities"] = 4
        self.write_report(self.report)
        with self.assertRaisesRegex(sources.SourceIndexError, "activity_count_mismatch"):
            self.build()

    def test_duplicate_kernel_rejected(self):
        self.report["coverage"]["activity_kernel_types"] *= 2
        self.write_report(self.report)
        with self.assertRaisesRegex(sources.SourceIndexError, "duplicate_kernel_name"):
            self.build()

    def test_duplicate_report_cannot_double_count(self):
        with self.assertRaisesRegex(sources.SourceIndexError, "duplicate_report"):
            self.build([self.report_path, self.report_path])

    def test_report_order_does_not_change_index(self):
        report = copy.deepcopy(self.report)
        report["coverage"]["kernel_activities"] = 7
        report["coverage"]["activity_kernel_types"][0]["activities"] = 7
        other = self.write_report(report, "other.json")
        first = self.build([self.report_path, other])
        second = self.build([other, self.report_path])
        self.assertEqual(first, second)
        self.assertEqual(first["coverage"]["kernel_activities"], 10)

    def test_paths_and_irrelevant_private_fields_do_not_serialize(self):
        self.report["private"] = {"pointer": "0xDEADBEEF", "path": str(self.root), "stream_handle": 1234}
        self.write_report(self.report)
        payload = json.dumps(self.build())
        self.assertNotIn("DEADBEEF", payload)
        self.assertNotIn(str(self.root), payload)
        self.assertNotIn("stream_handle", payload)

    def test_raw_pointer_kernel_name_rejected(self):
        self.report["coverage"]["activity_kernel_types"][0]["name"] = "kernel_0xDEADBEEF"
        self.write_report(self.report)
        with self.assertRaisesRegex(sources.SourceIndexError, "private_kernel_name"):
            self.build()

    def test_bad_counter_types_and_overflow_rejected(self):
        for value in (True, -1, 1.5, 1 << 64, "3"):
            with self.subTest(value=value):
                self.report["coverage"]["activity_kernel_types"][0]["activities"] = value
                self.write_report(self.report)
                with self.assertRaisesRegex(sources.SourceIndexError, "invalid_counter"):
                    self.build()

    def test_sum_overflow_rejected(self):
        self.report["coverage"]["activity_kernel_types"] = [{"name": "a", "activities": sources.MAX_U64}, {"name": "b", "activities": 1}]
        self.write_report(self.report)
        with self.assertRaisesRegex(sources.SourceIndexError, "counter_overflow"):
            self.build()

    def test_wrong_upstream_rejected(self):
        self.report["provenance"]["upstream_commit"] = "0" * 40
        self.write_report(self.report)
        with self.assertRaisesRegex(sources.SourceIndexError, "unpinned"):
            self.build()

    def test_malformed_and_duplicate_key_json_rejected(self):
        for value in (b"{", b"{\"x\":1,\"x\":2}", b"{\"x\": NaN}", b"\xff"):
            self.report_path.write_bytes(value)
            with self.subTest(value=value), self.assertRaises(sources.SourceIndexError):
                self.build()

    def test_excessive_json_integer_is_controlled_failure(self):
        self.report_path.write_bytes(b'{"counter":' + b'9' * 10000 + b'}')
        with self.assertRaises(sources.SourceIndexError):
            self.build()

    def test_oversize_report_rejected_with_bounded_read(self):
        with mock.patch.object(sources, "MAX_REPORT_BYTES", 8):
            with self.assertRaisesRegex(sources.SourceIndexError, "input_size_limit"):
                self.build()

    def test_too_many_reports_rejected_before_reads(self):
        with self.assertRaisesRegex(sources.SourceIndexError, "report_count_limit"):
            self.build([self.report_path] * (sources.MAX_REPORTS + 1))

    def test_invalid_report_shape_rejected(self):
        for value in ([], {}, {"schema_version": True, "report_type": "xvram.cuda_compat_audit"}):
            self.write_report(value)
            with self.subTest(value=value), self.assertRaises(sources.SourceIndexError):
                self.build()

    def test_source_fact_anchor_ambiguity_is_not_proof(self):
        fact = {"id": "fixture", "claim": "fixture", "anchors": [("cpy.cu", "cpy_scalar")], "obligation": "fixture"}
        with mock.patch.object(sources, "SOURCE_FACTS", (fact,)):
            result = sources.source_facts({self.source_path: self.source_text * 2})
        self.assertEqual(result[0]["status"], "unverified_source")
        self.assertIsNone(result[0]["claim"])

    def test_verified_source_fact_still_has_no_launch_binding(self):
        fact = {"id": "fixture", "claim": "fixture", "anchors": [("cpy.cu", "cpy_scalar")], "obligation": "fixture"}
        with mock.patch.object(sources, "SOURCE_FACTS", (fact,)):
            result = sources.source_facts({self.source_path: self.source_text})
        self.assertEqual(result[0]["status"], "source_proven_only")
        self.assertFalse(result[0]["captured_launch_binding"])

    def test_cli_refuses_overwrite_and_keeps_original(self):
        output = self.root / "index.json"
        output.write_text("original", encoding="utf-8")
        with mock.patch.object(sources, "build_index", return_value=self.build()), contextlib.redirect_stderr(io.StringIO()):
            code = sources.main(["--source-root", str(self.root), "--report", str(self.report_path), "--json", str(output)])
        self.assertEqual(code, 74)
        self.assertEqual(output.read_text(), "original")

    def test_cli_success_means_index_written_not_go(self):
        output = self.root / "index.json"
        with mock.patch.object(sources, "load_profile", return_value=(self.profile, "a" * 64)):
            code = sources.main(["--source-root", str(self.root), "--report", str(self.report_path), "--json", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["decision"]["verdict"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
