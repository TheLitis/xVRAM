"""Portable generator contract tests; compilation is owned by the native job."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from xvram import compat_audit_kernel_catalog as catalog
from xvram import compat_audit_kernel_templates as templates

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate-audit-kernel-header.py"
SPEC = importlib.util.spec_from_file_location("audit_kernel_header_generator", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)
CATALOG = Path(catalog.__file__).with_name("compat_audit_kernel_capture_catalog.json")


class HeaderGeneratorTests(unittest.TestCase):
    def test_portable_deterministic_header_and_supported_counts(self):
        value = generator.load_catalog(CATALOG)
        text = generator.render(value)
        self.assertEqual(text, generator.render(value))
        self.assertIn('#include "typed_capture.hpp"', text)
        self.assertIn("namespace xvram::launch_probe::typed_capture", text)
        self.assertIn("inline constexpr std::array<Kernel, 39> kernels", text)
        self.assertIn(generator.CATALOG_SHA256, text)
        self.assertEqual(text.count("}, true},"), 35)
        self.assertEqual(text.count("}, false},"), 4)
        self.assertIn("std::array<Argument, 0> arguments_36{}", text)

    def test_fields_expand_vectors_and_structs_without_padding(self):
        rope = catalog.expected_layout("rope_neox")
        self.assertEqual(generator.fields(rope[22]), [("value", "boolean", 0)])
        self.assertEqual(generator.fields(rope[17]), [("v0", "f32", 0), ("v1", "f32", 4)])
        quant = catalog.expected_layout("quantize_q8_1")
        self.assertEqual(generator.fields(quant[8]), [("x", "u32", 0), ("y", "u32", 4), ("z", "u32", 8)])
        softmax = generator.fields(catalog.expected_layout("soft_max_f32")[4])
        self.assertNotIn(12, [offset for _, _, offset in softmax])
        self.assertEqual(softmax[-1], ("m1", "f32", 124))
        fusion = generator.fields(catalog.expected_layout("mul_mat_vec_f")[3])
        self.assertEqual(sum(kind == "ptr" for _, kind, _ in fusion), 5)

    def test_wrong_abi_opaque_decoder_and_proof_escalation_rejected(self):
        value = generator.load_catalog(CATALOG)
        mutations = []
        changed = deepcopy(value)
        changed["kernels"][0]["arguments"][0]["size_bytes"] = 16
        mutations.append(changed)
        changed = deepcopy(value)
        changed["kernels"][35]["arguments"] = deepcopy(changed["kernels"][0]["arguments"])
        mutations.append(changed)
        changed = deepcopy(value)
        changed["kernels"][0]["memory_bounds_proven"] = True
        mutations.append(changed)
        changed = deepcopy(value)
        changed["schema_version"] = True
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(ValueError):
                generator.render(changed)

    def test_modified_bytes_rejected_even_if_still_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_bytes(CATALOG.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "unreviewed_hash"):
                generator.load_catalog(path)

    def test_duplicate_keys_and_oversized_input_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate_key"):
            generator.unique_object([("a", 1), ("a", 2)])
        with mock.patch.object(generator, "MAX_CATALOG_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "size_limit"):
                generator.load_catalog(CATALOG)

    def test_cli_output_stable_and_not_rewritten_when_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.hpp"
            argv = ["--catalog", str(CATALOG), "--output", str(output)]
            self.assertEqual(generator.main(argv), 0)
            before = output.stat().st_mtime_ns
            self.assertEqual(generator.main(argv), 0)
            self.assertEqual(output.stat().st_mtime_ns, before)
            self.assertEqual(output.read_text(), generator.render(generator.load_catalog(CATALOG)))


class TemplateProfileTests(unittest.TestCase):
    def test_all_custom_layouts_and_explicit_opaque_boundary(self):
        self.assertEqual(generator.CATALOG_SHA256, templates.CATALOG_SHA256)
        self.assertEqual(set(templates.PROFILES), set(range(1, 36)))
        for kid in range(1, 40):
            profile = templates.template_profile(kid, templates.CATALOG_SHA256)
            self.assertEqual(profile["supported"], kid < 36)
            if kid < 36:
                self.assertFalse(profile["binary_template_selection_proven"])
        self.assertEqual(templates.template_profile(34, templates.CATALOG_SHA256)["destination_bytes"], 2)
        self.assertEqual(templates.template_profile(35, templates.CATALOG_SHA256)["destination_bytes"], 4)

    def test_unknown_id_or_different_catalog_never_demangled(self):
        for kid in (0, 40, True, "1"):
            with self.assertRaises(ValueError):
                templates.template_profile(kid, templates.CATALOG_SHA256)
        with self.assertRaises(ValueError):
            templates.template_profile(1, "0"*64)


if __name__ == "__main__":
    unittest.main()
