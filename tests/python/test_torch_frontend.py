from __future__ import annotations

import unittest

from xvram.torch import InferenceRuntime, InferenceRuntimeError
from xvram.torch_runtime import InferenceBackend

try:
    import torch
except (ImportError, OSError):
    torch = None


class _Backend(InferenceBackend):
    def __init__(self) -> None:
        self.events = []
        self.input = None
        self.output = None

    def prepare(self, plan, state_provider):
        self.events.append(("prepare", plan.graph_hash, state_provider.targets()))

    def bind_input(self, value, host_tensor):
        self.events.append(("bind", value.name))
        self.input = host_tensor

    def prefetch(self, ranges):
        self.events.append(("prefetch", len(ranges)))

    def submit(self, request):
        self.events.append(("submit", request.sequence, request.node.target))
        if request.node.target == "aten.neg.default":
            self.output = -self.input
        elif request.node.target == "aten.mul.Tensor":
            self.output = self.input * 2
        else:  # pragma: no cover - the test models have one node
            raise RuntimeError("unexpected test node")
        return request.sequence

    def retire(self, token):
        self.events.append(("retire", token))

    def release_values(self, names):
        self.events.append(("release", names))

    def read_output(self, value):
        self.events.append(("read", value.name))
        return self.output

    def telemetry_snapshot(self):
        return {"events_recorded": 1, "events_retired": 1}

    def close(self):
        self.events.append(("close",))


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class TorchFrontendTests(unittest.TestCase):
    class _Neg(torch.nn.Module if torch is not None else object):
        def forward(self, value):
            return -value

    class _Mul(torch.nn.Module if torch is not None else object):
        def forward(self, value):
            return value * 2

    def runtime(self):
        backends = []

        def factory(_plan):
            backend = _Backend()
            backends.append(backend)
            return backend

        return InferenceRuntime(_torch_module=torch, _backend_factory=factory), backends

    def test_public_compile_is_static_callable_and_closes_once(self):
        runtime, backends = self.runtime()
        model = self._Neg().eval()
        example = torch.arange(4, dtype=torch.float32)

        compiled = runtime.compile_inference(model, (example,))
        result = compiled(example.clone())

        torch.testing.assert_close(result, -example)
        self.assertEqual(compiled.metadata.graph_hash, compiled.metadata.backend_hash)
        self.assertEqual(compiled.metadata.region_count, 1)
        self.assertEqual(compiled.telemetry()["events_retired"], 1)
        with self.assertRaisesRegex(InferenceRuntimeError, "already owns"):
            runtime.compile_inference(model, (example,))
        runtime.close()
        runtime.close()
        self.assertEqual(sum(event[0] == "close" for event in backends[0].events), 1)
        with self.assertRaisesRegex(InferenceRuntimeError, "no longer active"):
            compiled(example)

    def test_torch_compile_backend_validates_and_routes_cpu_input(self):
        runtime, _backends = self.runtime()
        model = self._Neg().eval()
        example = torch.arange(4, dtype=torch.float32)
        backend = runtime.backend(model, (example,))

        optimized = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        result = optimized(example)

        torch.testing.assert_close(result, -example)
        self.assertEqual(backend.backend_hash, backend.graph_hash)
        runtime.close()

    def test_torch_compile_backend_rejects_a_different_graph(self):
        runtime, _backends = self.runtime()
        example = torch.arange(4, dtype=torch.float32)
        backend = runtime.backend(self._Neg().eval(), (example,))
        optimized = torch.compile(
            self._Mul().eval(),
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        with self.assertRaisesRegex(Exception, "differs from strict export"):
            optimized(example)
        runtime.close()

    def test_cuda_materialization_is_bounded_by_reserved_headroom(self):
        backends = []

        def factory(_plan):
            backend = _Backend()
            backends.append(backend)
            return backend

        runtime = InferenceRuntime(
            device_headroom=1,
            _torch_module=torch,
            _backend_factory=factory,
        )
        example = torch.arange(4, dtype=torch.float32)
        compiled = runtime.compile_inference(self._Neg().eval(), (example,))
        with self.assertRaisesRegex(InferenceRuntimeError, "device headroom"):
            compiled.materialize_cuda(example)
        runtime.close()

    def test_public_runtime_exposes_opt_in_compression_configuration(self):
        runtime = InferenceRuntime(
            compression="adaptive",
            compression_codec="lz4",
            host_store_cap="12GiB",
            host_headroom="2GiB",
            compression_scratch_cap="128MiB",
            codec_slots=3,
            codec_workers=4,
            _torch_module=torch,
        )

        self.assertEqual(runtime.config.native_abi_version, 2)
        self.assertEqual(runtime.config.host_store_cap_bytes, 12 << 30)
        self.assertEqual(runtime.config.host_headroom_bytes, 2 << 30)
        self.assertEqual(runtime.config.compression_scratch_bytes, 128 << 20)
        runtime.close()


if __name__ == "__main__":
    unittest.main()
