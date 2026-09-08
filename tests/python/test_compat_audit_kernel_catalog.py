"""No-driver checks of the pinned typed-capture candidate catalog."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest import mock

from xvram import compat_audit_kernel_catalog as catalog
from xvram import compat_audit_kernel_ranges as ranges
from xvram.compat_audit_memory import MemoryRuleError


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(Path(catalog.__file__).with_name("compat_audit_kernel_capture_catalog.json").read_text())

    def test_all_reviewed_symbols_are_unique_and_opaque_never_decode(self):
        rows = self.catalog["kernels"]
        self.assertEqual(len(rows), 39)
        self.assertEqual(len({row["symbol"] for row in rows}), 39)
        self.assertEqual([row["catalog_id"] for row in rows], list(range(1, 40)))
        for row, family in zip(rows, catalog.BASELINE_FAMILIES):
            self.assertEqual(row["source_family"], family)
            self.assertFalse(row["source_semantics_bound_to_cubin"])
            self.assertFalse(row["memory_bounds_proven"])
            if family is None:
                self.assertEqual(row["status"], "unsupported_opaque_library")
                self.assertEqual(row["arguments"], [])
            else:
                self.assertEqual(row["arguments"], catalog.expected_layout(family))

    def test_nested_pointers_and_padding_cannot_be_raw_serialized(self):
        args = catalog.expected_layout("mul_mat_vec_f")
        fusion = args[3]
        self.assertEqual((fusion["offset_bytes"], fusion["size_bytes"], fusion["alignment_bytes"]), (24, 48, 8))
        self.assertEqual(fusion["serialization"], "fieldwise_no_padding")
        self.assertEqual([f["offset_bytes"] for f in fusion["fields"][:5]], [0, 8, 16, 24, 32])
        self.assertTrue(all(f["serialization"] == "resolved_allocation_reference" for f in fusion["fields"][:5]))

    def test_softmax_128_byte_struct_and_rope_bool_layout(self):
        softmax = catalog.expected_layout("soft_max_f32")[4]
        self.assertEqual((softmax["offset_bytes"], softmax["size_bytes"]), (32, 128))
        fields = {f["name"]: f for f in softmax["fields"]}
        self.assertEqual(fields["n_head_log2"]["offset_bytes"], 8)
        self.assertEqual(fields["ncols"]["offset_bytes"], 16)
        self.assertEqual(fields["m1"]["offset_bytes"], 124)
        rope = catalog.expected_layout("rope_neox")
        self.assertEqual((rope[17]["offset_bytes"], rope[17]["size_bytes"]), (84, 8))
        self.assertEqual((rope[22]["offset_bytes"], rope[22]["size_bytes"]), (116, 1))

    def test_unreviewed_baseline_rejected_before_external_access(self):
        with mock.patch.object(catalog, "_load", return_value=({}, "0"*64)), \
                mock.patch.object(catalog, "SourceKernelRanges") as source_verify:
            with self.assertRaisesRegex(MemoryRuleError, "unreviewed_kernel_catalog_baseline"):
                catalog.generate("unused", "unused", "unused", "unused")
            source_verify.assert_not_called()

    def test_generated_catalog_matches_source_layouts_and_is_deterministic(self):
        # The unit test supplies declaration fixtures, never native evidence.
        rows = self.catalog["kernels"]
        baseline = {"provenance": {"probe_trace": "a"*64}, "kernels": []}
        live = {"sha256": "a"*64, "layouts": []}
        import hashlib
        for row in rows:
            family = row["source_family"]
            # Opaque fixtures preserve metadata only and deliberately have no decoder.
            parameters = [{"offset_bytes": item["offset_bytes"], "size_bytes": item["size_bytes"]}
                          for item in row["arguments"]] if family else [{"offset_bytes": 0, "size_bytes": 8}]
            digest = hashlib.sha256(json.dumps([(p["offset_bytes"], p["size_bytes"]) for p in parameters],
                                              separators=(",", ":")).encode("ascii")).hexdigest()
            baseline["kernels"].append({"name": row["symbol"], "layout_sha256": digest,
                                       "live_layout_id": row["catalog_id"], "candidates": []})
            live["layouts"].append({"kernel_name": row["symbol"], "parameters": parameters})
        model = mock.Mock(provenance={"fixture": True}, extra_provenance={"fixture": True})
        with mock.patch.object(catalog, "_load", return_value=(baseline, catalog.BASELINE_SHA256)), \
                mock.patch.object(catalog, "validate_report"), \
                mock.patch.object(catalog, "SourceKernelRanges", return_value=model), \
                mock.patch.object(catalog, "analyze_trace", return_value=live):
            first = catalog.generate("unused", "unused", "unused", "unused")
            self.assertEqual(first, catalog.generate("unused", "unused", "unused", "unused"))
            changed = deepcopy(live)
            changed["layouts"][0]["parameters"][0]["size_bytes"] = 16
            with mock.patch.object(catalog, "analyze_trace", return_value=changed):
                with self.assertRaisesRegex(MemoryRuleError, "layout_hash_mismatch"):
                    catalog.generate("unused", "unused", "unused", "unused")


class DependencyManifestTests(unittest.TestCase):
    def test_manifest_has_exact_commit_and_unique_dependencies(self):
        profile = json.loads(Path(ranges.__file__).with_name("compat_audit_kernel_source_profile.json").read_text())
        self.assertEqual(profile["upstream_commit"], ranges.sources.COMMIT)
        self.assertEqual(len(profile["files"]), 10)
        self.assertEqual(len({row["path"] for row in profile["files"]}), 10)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in profile["files"]))

    def test_failed_dependency_hash_and_missing_optional_root_fail_closed(self):
        with mock.patch.object(ranges.sources, "verify_sources", return_value=({}, [{"status": "hash_mismatch"}])):
            with self.assertRaisesRegex(MemoryRuleError, "required_kernel_dependency_unverified"):
                ranges.verify_extra_sources("unused")
        model = object.__new__(ranges.SourceKernelRanges)
        model.extra_provenance = None
        with self.assertRaisesRegex(MemoryRuleError, "required_kernel_dependency_unverified"):
            model.evaluate("mmq_stream_k")


if __name__ == "__main__":
    unittest.main()
