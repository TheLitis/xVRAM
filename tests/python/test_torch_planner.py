from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from xvram.torch_planner import (  # noqa: E402
    AccessMode,
    PlanError,
    build_inference_plan,
    backend_operator_targets,
    embedding_row_ranges,
    operator_allowlist_hash,
    supported_operator_targets,
)


class _Storage:
    def __init__(self, pointer, size):
        self._pointer = pointer
        self._size = size

    def data_ptr(self):
        return self._pointer

    def nbytes(self):
        return self._size


class _Tensor:
    def __init__(
        self,
        sizes,
        *,
        dtype="torch.float32",
        element_size=4,
        strides=None,
        offset=0,
        pointer=100,
        storage_size=None,
        data=None,
    ):
        self._sizes = tuple(sizes)
        self._strides = tuple(strides or _dense_strides(sizes))
        self._element_size = element_size
        self._offset = offset
        self.dtype = dtype
        self.device = SimpleNamespace(type="cpu")
        try:
            span = (
                sum((size - 1) * stride for size, stride in zip(sizes, self._strides)) + 1
            ) * element_size
        except TypeError:
            span = element_size
        self._storage = _Storage(pointer, storage_size or offset * element_size + span)
        self._data = data

    def size(self):
        return self._sizes

    def stride(self):
        return self._strides

    def element_size(self):
        return self._element_size

    def storage_offset(self):
        return self._offset

    def untyped_storage(self):
        return self._storage

    def detach(self):
        return self

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return self._data


def _dense_strides(sizes):
    stride = 1
    result = []
    for size in reversed(tuple(sizes)):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


class _Node:
    def __init__(self, name, op, target=None, args=(), inputs=(), tensor=None, kwargs=None):
        self.name = name
        self.op = op
        self.target = target if target is not None else op
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.all_input_nodes = tuple(inputs)
        self.meta = {} if tensor is None else {"val": tensor}


def _input_spec(name, kind, target=None):
    return SimpleNamespace(
        arg=SimpleNamespace(name=name), kind=SimpleNamespace(name=kind), target=target
    )


def _output_spec(name):
    return SimpleNamespace(
        arg=SimpleNamespace(name=name), kind=SimpleNamespace(name="USER_OUTPUT"), target=None
    )


class _Exported:
    def __init__(self, nodes, inputs, outputs, state=None):
        self.graph_module = SimpleNamespace(graph=SimpleNamespace(nodes=tuple(nodes)))
        self.graph_signature = SimpleNamespace(
            input_specs=tuple(inputs), output_specs=tuple(outputs)
        )
        self.state_dict = dict(state or {})
        self.constants = {}


def _embedding_linear_program(pointer=100):
    embedding_state = _Tensor((8, 4), pointer=pointer, storage_size=128)
    linear_state = _Tensor((3, 4), pointer=pointer + 1, storage_size=48)
    weight = _Node("p_embedding", "placeholder", tensor=_Tensor((8, 4)))
    projection = _Node("p_projection", "placeholder", tensor=_Tensor((3, 4)))
    tokens = _Node(
        "tokens",
        "placeholder",
        tensor=_Tensor((4,), dtype="torch.int64", element_size=8),
    )
    embedding = _Node(
        "embedding",
        "call_function",
        "aten.embedding.default",
        args=(weight, tokens, -1, False, False),
        inputs=(weight, tokens),
        tensor=_Tensor((4, 4)),
    )
    linear = _Node(
        "linear",
        "call_function",
        "aten.linear.default",
        args=(embedding, projection, None),
        inputs=(embedding, projection),
        tensor=_Tensor((4, 3)),
    )
    output = _Node("output", "output", args=((linear,),), inputs=(linear,))
    return _Exported(
        (weight, projection, tokens, embedding, linear, output),
        (
            _input_spec("p_embedding", "PARAMETER", "embedding.weight"),
            _input_spec("p_projection", "PARAMETER", "projection.weight"),
            _input_spec("tokens", "USER_INPUT"),
        ),
        (_output_spec("linear"),),
        {"embedding.weight": embedding_state, "projection.weight": linear_state},
    )


class TorchPlannerTests(unittest.TestCase):
    def test_v1_frontend_and_backend_operator_contract_is_explicit(self):
        frontend = set(supported_operator_targets())
        for target in (
            "aten.mm.default",
            "aten.addmm.default",
            "aten.bmm.default",
            "aten.clone.default",
            "aten._to_copy.default",
            "aten.permute.default",
            "operator.getitem",
        ):
            self.assertIn(target, frontend)
        backend = backend_operator_targets()
        self.assertEqual(backend["mm_out"], "aten.mm.out")
        self.assertEqual(backend["addmm_out"], "aten.addmm.out")
        self.assertEqual(backend["bmm_out"], "aten.bmm.out")
        self.assertEqual(backend["cast_copy_out"], "aten.copy.out")

    def test_embedding_plan_is_strict_and_hash_is_address_independent(self):
        first = build_inference_plan(_embedding_linear_program(pointer=100))
        second = build_inference_plan(_embedding_linear_program(pointer=9000))

        self.assertEqual(first.graph_hash, second.graph_hash)
        self.assertTrue(first.is_hash_equivalent(second))
        self.assertEqual(first.allowlist_hash, operator_allowlist_hash())
        self.assertTrue(first.no_fallback)
        self.assertEqual(first.input_names, ("tokens",))
        self.assertEqual(first.output_names, ("linear",))
        self.assertEqual(first.nodes[0].adapter, "embedding_out")
        self.assertIsNotNone(first.nodes[0].embedding_range)
        self.assertFalse(
            any("p_embedding" in item.values for item in first.nodes[0].accesses)
        )

    def test_tied_state_uses_one_logical_storage(self):
        shared_a = _Tensor((4, 4), pointer=77, storage_size=64)
        shared_b = _Tensor((4, 4), pointer=77, storage_size=64)
        weight_a = _Node("p_a", "placeholder", tensor=_Tensor((4, 4)))
        weight_b = _Node("p_b", "placeholder", tensor=_Tensor((4, 4)))
        value = _Node("x", "placeholder", tensor=_Tensor((2, 4)))
        left = _Node(
            "left",
            "call_function",
            "aten.linear.default",
            args=(value, weight_a, None),
            inputs=(value, weight_a),
            tensor=_Tensor((2, 4)),
        )
        right = _Node(
            "right",
            "call_function",
            "aten.linear.default",
            args=(value, weight_b, None),
            inputs=(value, weight_b),
            tensor=_Tensor((2, 4)),
        )
        added = _Node(
            "added",
            "call_function",
            "aten.add.Tensor",
            args=(left, right),
            inputs=(left, right),
            tensor=_Tensor((2, 4)),
        )
        output = _Node("output", "output", args=((added,),), inputs=(added,))
        exported = _Exported(
            (weight_a, weight_b, value, left, right, added, output),
            (
                _input_spec("p_a", "PARAMETER", "a"),
                _input_spec("p_b", "PARAMETER", "b"),
                _input_spec("x", "USER_INPUT"),
            ),
            (_output_spec("added"),),
            {"a": shared_a, "b": shared_b},
        )

        plan = build_inference_plan(exported)
        self.assertEqual(plan.value("p_a").storage_id, plan.value("p_b").storage_id)
        self.assertEqual(plan.logical_state_bytes, 64)

    def test_alias_keeps_root_slot_live_until_last_alias_use(self):
        state = _Tensor((4, 4), pointer=50, storage_size=64)
        weight = _Node("p_w", "placeholder", tensor=_Tensor((4, 4)))
        value = _Node("x", "placeholder", tensor=_Tensor((2, 4)))
        linear = _Node(
            "linear",
            "call_function",
            "aten.linear.default",
            args=(value, weight, None),
            inputs=(value, weight),
            tensor=_Tensor((2, 4)),
        )
        viewed = _Node(
            "viewed",
            "call_function",
            "aten.view.default",
            args=(linear, (1, 2, 4)),
            inputs=(linear,),
            tensor=_Tensor((1, 2, 4)),
        )
        negated = _Node(
            "negated",
            "call_function",
            "aten.neg.default",
            args=(viewed,),
            inputs=(viewed,),
            tensor=_Tensor((1, 2, 4)),
        )
        output = _Node("output", "output", args=((negated,),), inputs=(negated,))
        exported = _Exported(
            (weight, value, linear, viewed, negated, output),
            (
                _input_spec("p_w", "PARAMETER", "weight"),
                _input_spec("x", "USER_INPUT"),
            ),
            (_output_spec("negated"),),
            {"weight": state},
        )

        plan = build_inference_plan(exported)
        self.assertEqual(plan.value("viewed").alias_of, "linear")
        self.assertEqual(plan.value("linear").last_use_index, 4)
        self.assertNotEqual(
            plan.value("linear").activation_slot,
            plan.value("negated").activation_slot,
        )

    def test_embedding_rows_are_sorted_deduplicated_and_merged(self):
        ranges = embedding_row_ranges(
            (5, 2, 3, 2, 4, 8),
            vocabulary_rows=10,
            row_bytes=16,
            storage_id="weights",
            storage_offset_bytes=32,
        )
        self.assertEqual(
            [(item.offset_bytes, item.length_bytes, item.mode) for item in ranges],
            [
                (64, 64, AccessMode.READ),
                (160, 16, AccessMode.READ),
            ],
        )
        with self.assertRaises(IndexError):
            embedding_row_ranges(
                (10,), vocabulary_rows=10, row_bytes=16, storage_id="weights"
            )

    def test_unsupported_operator_and_dynamic_metadata_are_rejected(self):
        value = _Node("x", "placeholder", tensor=_Tensor((2, 4)))
        result = _Node(
            "relu",
            "call_function",
            "aten.relu.default",
            args=(value,),
            inputs=(value,),
            tensor=_Tensor((2, 4)),
        )
        output = _Node("output", "output", args=((result,),), inputs=(result,))
        exported = _Exported(
            (value, result, output),
            (_input_spec("x", "USER_INPUT"),),
            (_output_spec("relu"),),
        )
        with self.assertRaisesRegex(PlanError, "allowlist"):
            build_inference_plan(exported)

        class _Symbolic:
            def __int__(self):
                raise ValueError("symbolic")

        symbolic = _Node(
            "x", "placeholder", tensor=_Tensor((_Symbolic(), 4), strides=(4, 1))
        )
        output = _Node("output", "output", args=((symbolic,),), inputs=(symbolic,))
        exported = _Exported(
            (symbolic, output),
            (_input_spec("x", "USER_INPUT"),),
            (_output_spec("x"),),
        )
        with self.assertRaisesRegex(PlanError, "static integer"):
            build_inference_plan(exported)


if __name__ == "__main__":
    unittest.main()
