from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_torch_planner import _Tensor, _embedding_linear_program  # noqa: E402
from xvram.torch_planner import build_inference_plan  # noqa: E402
from xvram.torch_runtime import (  # noqa: E402
    CpuModuleStateProvider,
    InferenceBackend,
    InferenceBackendFactory,
    InferenceRuntime,
    RuntimeState,
    compile_inference,
)


class _Module:
    training = False

    def __init__(self):
        exported = _embedding_linear_program()
        self.embedding = exported.state_dict["embedding.weight"]
        self.projection = exported.state_dict["projection.weight"]

    def named_parameters(self, **_kwargs):
        return (
            ("embedding.weight", self.embedding),
            ("projection.weight", self.projection),
        )

    def named_buffers(self, **_kwargs):
        return ()


class _RecordingBackend(InferenceBackend):
    def __init__(self, *, null_token=False, fail_prepare=False):
        self.events = []
        self.requests = []
        self.null_token = null_token
        self.fail_prepare = fail_prepare

    def prepare(self, plan, state_provider):
        self.events.append(("prepare", plan.graph_hash, state_provider.targets()))
        if self.fail_prepare:
            raise RuntimeError("injected prepare failure")

    def bind_input(self, value, host_tensor):
        self.events.append(("bind", value.name, tuple(host_tensor.size())))

    def prefetch(self, ranges):
        self.events.append(("prefetch", ranges))

    def submit(self, request):
        self.requests.append(request)
        self.events.append(("submit", request.sequence, request.node.name))
        return None if self.null_token else "generation-{}".format(request.sequence)

    def retire(self, token):
        self.events.append(("retire", token))

    def release_values(self, names):
        self.events.append(("release", names))

    def read_output(self, value):
        self.events.append(("read", value.name))
        return {"output": value.name}

    def close(self):
        self.events.append(("close",))


class _Factory(InferenceBackendFactory):
    def __init__(self):
        self.hashes = []
        self.backend = _RecordingBackend()

    def create(self, plan):
        self.hashes.append((plan.graph_hash, plan.allowlist_hash))
        return self.backend


class TorchRuntimeTests(unittest.TestCase):
    def _runtime(self, backend=None):
        exported = _embedding_linear_program()
        module = _Module()
        provider = CpuModuleStateProvider(module)
        plan = build_inference_plan(exported, state_provider=provider)
        selected = backend or _RecordingBackend()
        return InferenceRuntime(plan, selected, provider), selected

    def test_scheduler_retires_every_generation_before_release(self):
        runtime, backend = self._runtime()
        tokens = _Tensor(
            (4,),
            dtype="torch.int64",
            element_size=8,
            data=[1, 2, 3, 5],
        )

        output = runtime.run(tokens)

        self.assertEqual(output, {"output": "linear"})
        self.assertEqual(runtime.state, RuntimeState.READY)
        self.assertEqual(runtime.run_count, 1)
        self.assertEqual([item.node.name for item in backend.requests], ["embedding", "linear"])
        event_names = [item[0] for item in backend.events]
        self.assertLess(event_names.index("retire"), event_names.index("release"))

        embedding_request = backend.requests[0]
        state_ranges = [
            item for item in embedding_request.ranges if item.storage_id == "state:0"
        ]
        self.assertEqual(
            [(item.offset_bytes, item.length_bytes) for item in state_ranges],
            [(16, 48), (80, 16)],
        )

    def test_null_generation_token_poison_runtime(self):
        runtime, backend = self._runtime(_RecordingBackend(null_token=True))
        tokens = _Tensor(
            (4,), dtype="torch.int64", element_size=8, data=[1, 2, 3, 5]
        )
        with self.assertRaisesRegex(RuntimeError, "null generation"):
            runtime.run(tokens)
        self.assertEqual(runtime.state, RuntimeState.FAILED)
        self.assertFalse(any(item[0] == "release" for item in backend.events))
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            runtime.run(tokens)

    def test_static_input_contract_is_checked_before_submission(self):
        runtime, backend = self._runtime()
        wrong = _Tensor(
            (3,), dtype="torch.int64", element_size=8, data=[1, 2, 3]
        )
        with self.assertRaisesRegex(ValueError, "static export metadata"):
            runtime.run(wrong)
        self.assertFalse(backend.requests)

    def test_close_is_idempotent(self):
        runtime, backend = self._runtime()
        runtime.prepare()
        runtime.close()
        runtime.close()
        self.assertEqual(runtime.state, RuntimeState.CLOSED)
        self.assertEqual(sum(item[0] == "close" for item in backend.events), 1)

    def test_failed_prepare_still_enters_backend_cleanup_boundary(self):
        runtime, backend = self._runtime(_RecordingBackend(fail_prepare=True))
        with self.assertRaisesRegex(RuntimeError, "injected prepare"):
            runtime.prepare()
        self.assertEqual(runtime.state, RuntimeState.FAILED)
        runtime.close()
        self.assertEqual(runtime.state, RuntimeState.CLOSED)
        self.assertEqual(sum(item[0] == "close" for item in backend.events), 1)

    def test_compile_inference_forces_strict_export(self):
        exported = _embedding_linear_program()
        calls = []

        class _ExportNamespace:
            @staticmethod
            def export(module, args, kwargs, strict):
                calls.append((module, args, kwargs, strict))
                return exported

        module = _Module()
        backend = _RecordingBackend()
        runtime = compile_inference(
            module,
            (
                _Tensor(
                    (4,),
                    dtype="torch.int64",
                    element_size=8,
                    data=[1, 2, 3, 5],
                ),
            ),
            backend=backend,
            torch_module=type("_Torch", (), {"export": _ExportNamespace}),
        )
        self.assertIsInstance(runtime, InferenceRuntime)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][3])

    def test_backend_factory_receives_hash_stable_plan(self):
        exported = _embedding_linear_program()
        calls = []

        class _ExportNamespace:
            @staticmethod
            def export(module, args, kwargs, strict):
                calls.append(strict)
                return exported

        factory = _Factory()
        runtime = compile_inference(
            _Module(),
            (_Tensor((4,), dtype="torch.int64", element_size=8, data=[1, 2, 3, 5]),),
            backend_factory=factory,
            torch_module=type("_Torch", (), {"export": _ExportNamespace}),
        )
        self.assertEqual(factory.hashes, [(runtime.plan.graph_hash, runtime.plan.allowlist_hash)])
        with self.assertRaisesRegex(TypeError, "exactly one"):
            compile_inference(
                _Module(),
                (_Tensor((4,), dtype="torch.int64", element_size=8, data=[1, 2, 3, 5]),),
                backend=factory.backend,
                backend_factory=factory,
                torch_module=type("_Torch", (), {"export": _ExportNamespace}),
            )


if __name__ == "__main__":
    unittest.main()
