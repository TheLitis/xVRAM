from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from xvram.lifetime import (  # noqa: E402
    TensorLifetime,
    TensorRole,
    classify_module_tensors,
    classify_tensor,
)


class _FakeModule:
    def named_parameters(self, **_kwargs):
        return [("encoder.weight", object()), ("encoder.bias", object())]

    def named_buffers(self, **_kwargs):
        return [("running_mean", object())]


class _IncompleteModule:
    def named_parameters(self, **_kwargs):
        return []


class LifetimeTests(unittest.TestCase):
    def test_unknown_tensor_is_not_inferred(self):
        tensor = type("Tensor", (), {"requires_grad": True, "is_leaf": True})()
        hint = classify_tensor(tensor)

        self.assertEqual(hint.lifetime, TensorLifetime.UNKNOWN)
        self.assertEqual(hint.role, TensorRole.UNKNOWN)
        self.assertFalse(hint.explicit)
        self.assertTrue(hint.advisory_only)

    def test_explicit_roles_have_conservative_lifetimes(self):
        self.assertEqual(classify_tensor(role="input").lifetime, TensorLifetime.EXTERNAL)
        self.assertEqual(classify_tensor(role="parameter").lifetime, TensorLifetime.PERSISTENT)
        self.assertEqual(classify_tensor(role="activation").lifetime, TensorLifetime.ITERATION)
        self.assertEqual(classify_tensor(role="temporary").lifetime, TensorLifetime.EPHEMERAL)

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown tensor role"):
            classify_tensor(role="probably-short-lived")

    def test_registered_module_tensors_are_persistent(self):
        hints = classify_module_tensors(_FakeModule())

        self.assertEqual(set(hints), {"encoder.weight", "encoder.bias", "running_mean"})
        self.assertEqual(hints["encoder.weight"].role, TensorRole.PARAMETER)
        self.assertEqual(hints["running_mean"].role, TensorRole.BUFFER)
        self.assertTrue(all(hint.lifetime is TensorLifetime.PERSISTENT for hint in hints.values()))

    def test_incomplete_module_contract_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "named_buffers"):
            classify_module_tensors(_IncompleteModule())


if __name__ == "__main__":
    unittest.main()
