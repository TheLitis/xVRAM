"""Standard unittest discovery with filename exclusions applied before import.

The compatibility audit has its own CTest suite and timeout. Its modules must
not also run (or be imported) in the broader Python suite. Keep the inclusion
pattern broad so unrelated and future flat modules, including test_fx/test_lifetime,
remain covered without maintaining an allowlist of subsystem prefixes. Test
packages, invalid discovery names and custom load_tests hooks are not supported.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
from pathlib import Path
import re
import unittest


def matches_test_file(name, pattern, exclusions=()):
    return fnmatch.fnmatch(name, pattern) and not any(fnmatch.fnmatch(name, item) for item in exclusions)


def validate_flat_test_directory(start_dir, pattern):
    """Fail before discovery when test layout exceeds the supported flat policy.

    The source check rejects declared/imported hooks without importing test
    modules. The loader also guards the resulting module namespace so a computed
    hook cannot be invoked by this runner. This is a test convention, not a
    sandbox for arbitrary Python code or custom import machinery.
    """
    directory = Path(start_dir)
    files = []
    for path in sorted(directory.iterdir()):
        if path.name == "__pycache__" and path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink():
            raise ValueError("test directory symlinks are unsupported")
        if path.is_dir():
            raise ValueError("nested test directories are unsupported")
        if path.name == "__init__.py":
            raise ValueError("test packages are unsupported")
        if path.suffix != ".py":
            continue
        if matches_test_file(path.name, pattern):
            if not re.fullmatch(r"[a-zA-Z_]\w*\.py", path.name) or not path.stem.isidentifier():
                raise ValueError("invalid unittest discovery filename")
            files.append(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            identifier = node.id if isinstance(node, ast.Name) else getattr(node, "name", None)
            if identifier == "load_tests" or (isinstance(node, ast.alias) and node.asname == "load_tests") or (
                    isinstance(node, ast.Constant) and node.value == "load_tests"):
                raise ValueError("custom load_tests hooks are unsupported")
    return files


class ExcludingLoader(unittest.TestLoader):
    def __init__(self, exclusions=()):
        super().__init__()
        self.exclusions = tuple(exclusions)

    def _match_path(self, path, full_path, pattern):
        # unittest's discovery hook is evaluated before module import. Delegating
        # the positive match preserves its normal filesystem/discovery semantics.
        return super()._match_path(path, full_path, pattern) and matches_test_file(path, pattern, self.exclusions)

    def loadTestsFromModule(self, module, *, pattern=None):
        if getattr(module, "load_tests", None) is not None:
            raise ValueError("custom load_tests hooks are unsupported")
        return super().loadTestsFromModule(module, pattern=pattern)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-dir", required=True)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        validate_flat_test_directory(args.start_dir, args.pattern)
    except (OSError, SyntaxError, ValueError) as error:
        parser.error(str(error))
    suite = ExcludingLoader(args.exclude).discover(args.start_dir, pattern=args.pattern)
    if suite.countTestCases() == 0:
        parser.error("test selection is empty")
    return 0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
