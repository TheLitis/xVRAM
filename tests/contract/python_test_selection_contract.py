"""CPU-only partition proof for the supported flat, ordinary-unittest layout."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests" / "helpers" / "run_python_tests.py"
TEST_DIRECTORY = ROOT / "tests" / "python"
PATTERN = "test_*.py"
AUDIT_PATTERN = "test_compat_audit*.py"
spec = importlib.util.spec_from_file_location("xvram_test_selection_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PythonTestSelectionContract(unittest.TestCase):
    def partition(self, names):
        all_tests = {name for name in names if runner.matches_test_file(Path(name).name, PATTERN)}
        audit = {name for name in names if runner.matches_test_file(Path(name).name, AUDIT_PATTERN)}
        other = {name for name in names if runner.matches_test_file(Path(name).name, PATTERN, (AUDIT_PATTERN,))}
        self.assertFalse(audit & other, "Python suites overlap")
        self.assertEqual(all_tests, audit | other, "Python suites drop discovered test modules")
        return audit, other

    def test_current_files_are_disjoint_exhaustive_and_include_legacy_names(self):
        files = runner.validate_flat_test_directory(TEST_DIRECTORY, PATTERN)
        self.assertTrue(files)
        audit, other = self.partition([str(path.relative_to(TEST_DIRECTORY)) for path in files])
        self.assertTrue(audit)
        self.assertTrue(other)
        self.assertIn("test_fx.py", other)
        self.assertIn("test_lifetime.py", other)

    def test_future_names_remain_partitioned_without_prefix_allowlist(self):
        audit, other = self.partition(["test_future_feature.py", "test_fx.py", "test_lifetime.py",
                                       "test_compat_audit_future.py", "test_compat_audit.py", "helper.py"])
        self.assertEqual(audit, {"test_compat_audit_future.py", "test_compat_audit.py"})
        self.assertEqual(other, {"test_future_feature.py", "test_fx.py", "test_lifetime.py"})

    def test_invalid_filename_is_rejected_instead_of_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "test_feature-extra.py").write_text("# invalid discovery name\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid unittest discovery filename"):
                runner.validate_flat_test_directory(directory, PATTERN)
            self.assertNotEqual(self.invoke(directory).returncode, 0)

    def test_nested_package_and_root_package_are_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nested = directory / "package"
            nested.mkdir()
            (nested / "__init__.py").write_text("", encoding="utf-8")
            (nested / "test_nested.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nested test directories"):
                runner.validate_flat_test_directory(directory, PATTERN)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "__init__.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test packages"):
                runner.validate_flat_test_directory(directory, PATTERN)

    def test_custom_discovery_hooks_are_rejected_before_use(self):
        declarations = ("def load_tests(loader, tests, pattern):\n    return tests\n",
                        "from helpers import custom as load_tests\n",
                        "from helpers import load_tests\n",
                        "load_tests = lambda *args: None\n",
                        "globals()['load_tests'] = lambda *args: None\n")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for declaration in declarations:
                with self.subTest(declaration=declaration):
                    (directory / "test_hook.py").write_text(declaration, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "custom load_tests hooks"):
                        runner.validate_flat_test_directory(directory, PATTERN)
                    self.assertNotEqual(self.invoke(directory).returncode, 0)
            # A computed hook bypasses the static source convention but is still
            # rejected before unittest can invoke the imported module's hook.
            (directory / "test_hook.py").write_text(
                "globals()['load_' + 'tests'] = lambda *args: (_ for _ in ()).throw(RuntimeError('hook was invoked'))\n",
                encoding="utf-8")
            result = self.invoke(directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("custom load_tests hooks are unsupported", result.stderr)
            self.assertNotIn("RuntimeError: hook was invoked", result.stderr)

    def invoke(self, directory, *extra):
        return subprocess.run([sys.executable, str(RUNNER), "--start-dir", str(directory),
                               "--pattern", PATTERN, "--exclude", AUDIT_PATTERN, *extra],
                              capture_output=True, text=True, timeout=10, check=False)

    def test_excluded_modules_are_not_imported_and_plain_names_still_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "test_compat_audit_excluded.py").write_text("raise RuntimeError('excluded module imported')\n", encoding="utf-8")
            passing = "import unittest\nclass Case(unittest.TestCase):\n    def test_selected(self):\n        self.assertTrue(True)\n"
            for name in ("test_fx.py", "test_lifetime.py", "test_future_feature.py"):
                (directory / name).write_text(passing, encoding="utf-8")
            result = self.invoke(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Ran 3 tests", result.stderr)
            self.assertNotIn("excluded module imported", result.stderr)

    def test_selected_failure_and_empty_suite_do_not_report_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertNotEqual(self.invoke(directory).returncode, 0)
            (directory / "test_future_failure.py").write_text(
                "import unittest\nclass Case(unittest.TestCase):\n    def test_failure(self):\n        self.fail('selected failure')\n",
                encoding="utf-8")
            result = self.invoke(directory)
            self.assertEqual(result.returncode, 1)
            self.assertIn("selected failure", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=PATTERN)
    parser.add_argument("--audit-pattern", default=AUDIT_PATTERN)
    args = parser.parse_args()
    PATTERN, AUDIT_PATTERN = args.pattern, args.audit_pattern
    unittest.main(argv=[sys.argv[0]])
