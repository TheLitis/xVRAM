from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_torch_planner import (  # noqa: E402
    _Exported,
    _Node,
    _Tensor,
    _embedding_linear_program,
    _input_spec,
    _output_spec,
)
from xvram.torch_native import (  # noqa: E402
    NativeInferenceBackend,
    NativeLeaseInfo,
    NativeRuntimeConfig,
    NativeTorchRuntimeError,
    _ApiV1,
    _SessionConfigV1,
    _TelemetryV1,
    find_runtime_library,
)
from xvram.torch_planner import (  # noqa: E402
    StateTensorInfo,
    TensorSpec,
    build_inference_plan,
    describe_cpu_state_tensor,
)
from xvram.torch_runtime import (  # noqa: E402
    BackendContractError,
    ExecutionRequest,
    InferenceRuntime,
    StateProvider,
)


class _Provider(StateProvider):
    def __init__(self, exported):
        self._state = dict(exported.state_dict)

    def describe(self, target):
        return describe_cpu_state_tensor(target, self._state[target])

    def targets(self):
        return tuple(sorted(self._state))


class _ByteTensor:
    dtype = "torch.uint8"
    device = SimpleNamespace(type="cpu")

    def __init__(self, pointer, length):
        self._pointer = int(pointer)
        self._length = int(length)

    def data_ptr(self):
        return self._pointer

    def numel(self):
        return self._length

    def element_size(self):
        return 1

    def is_contiguous(self):
        return True


class _ChunkSource:
    def __init__(self, target, pointer):
        self.target = target
        self.pointer = int(pointer)

    def iter_chunks(self, chunk_bytes):
        del chunk_bytes
        yield SimpleNamespace(
            target=self.target,
            offset_bytes=0,
            data=_ByteTensor(self.pointer, 16),
        )
        yield SimpleNamespace(
            target=self.target,
            offset_bytes=16,
            data=_ByteTensor(self.pointer + 16, 16),
        )


class _ChunkedProvider(StateProvider):
    def __init__(self):
        dense = TensorSpec(
            dtype="torch.float32",
            element_size=4,
            sizes=(2, 4),
            strides=(4, 1),
            storage_offset_elements=0,
            span_bytes=32,
            numel=8,
        )
        shifted = TensorSpec(
            dtype="torch.float32",
            element_size=4,
            sizes=(2, 4),
            strides=(4, 1),
            storage_offset_elements=8,
            span_bytes=32,
            numel=8,
        )
        identity = ("chunked", 1)
        self._infos = {
            "a": StateTensorInfo(
                "a", _ChunkSource("a", 40000), identity, 64, 0, dense
            ),
            "b": StateTensorInfo(
                "b", _ChunkSource("b", 50000), identity, 64, 32, shifted
            ),
        }

    def describe(self, target):
        return self._infos[target]

    def targets(self):
        return ("a", "b")


def _chunked_tied_program():
    weight_a = _Node("p_a", "placeholder", tensor=_Tensor((2, 4)))
    weight_b = _Node("p_b", "placeholder", tensor=_Tensor((2, 4)))
    value = _Node("x", "placeholder", tensor=_Tensor((2, 4)))
    left = _Node(
        "left",
        "call_function",
        "aten.linear.default",
        args=(value, weight_a, None),
        inputs=(value, weight_a),
        tensor=_Tensor((2, 2)),
    )
    right = _Node(
        "right",
        "call_function",
        "aten.linear.default",
        args=(value, weight_b, None),
        inputs=(value, weight_b),
        tensor=_Tensor((2, 2)),
    )
    added = _Node(
        "added",
        "call_function",
        "aten.add.Tensor",
        args=(left, right),
        inputs=(left, right),
        tensor=_Tensor((2, 2)),
    )
    output = _Node("output", "output", args=((added,),), inputs=(added,))
    return _Exported(
        (weight_a, weight_b, value, left, right, added, output),
        (
            _input_spec("p_a", "PARAMETER", "a"),
            _input_spec("p_b", "PARAMETER", "b"),
            _input_spec("x", "USER_INPUT"),
        ),
        (_output_spec("added"),),
    )


class _HostOutput:
    device = SimpleNamespace(type="cpu")

    def __init__(self, sizes, strides, dtype, pointer):
        self._sizes = tuple(sizes)
        self._strides = tuple(strides)
        self.dtype = dtype
        self._pointer = pointer

    def size(self):
        return self._sizes

    def stride(self):
        return self._strides

    def data_ptr(self):
        return self._pointer


class _Context:
    def __init__(self, events, enter, leave):
        self.events = events
        self.enter = enter
        self.leave = leave

    def __enter__(self):
        self.events.append(self.enter)
        return self

    def __exit__(self, *_args):
        self.events.append(self.leave)
        return False


class _ManagedView:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<view {}>".format(self.name)


class _Op:
    def __init__(self, events, name, *, fail=False):
        self.events = events
        self.name = name
        self.fail = fail

    def __call__(self, *args, **kwargs):
        self.events.append(("op", self.name, args, dict(kwargs)))
        if self.fail:
            raise RuntimeError("injected ATen submission failure")
        return kwargs.get("out", _ManagedView("scratch:" + self.name))


class _ScratchAllocator:
    def allocator(self):
        return "allocator-capsule"


class _Pool:
    def __init__(self, events, allocator=None, no_split=False):
        self.events = events
        self.events.append(("pool", allocator, no_split))

    def __del__(self):
        self.events.append(("pool_destroy",))


class _FakeCuda:
    def __init__(self, events):
        self.events = events
        self._device = -1
        self.memory = SimpleNamespace(CUDAPluggableAllocator=None)

    def is_available(self):
        return True

    def set_device(self, device):
        self._device = int(device)
        self.events.append(("set_device", int(device)))

    def init(self):
        self.events.append(("cuda_init",))

    def current_device(self):
        return self._device

    def is_current_stream_capturing(self):
        return False

    def ExternalStream(self, identity, device=None):
        self.events.append(("external_stream", int(identity), int(device)))
        return SimpleNamespace(identity=int(identity), device=int(device))

    def stream(self, stream):
        return _Context(
            self.events,
            ("stream_enter", stream.identity),
            ("stream_exit", stream.identity),
        )

    def MemPool(self, allocator=None, no_split=False):
        return _Pool(self.events, allocator=allocator, no_split=no_split)

    @contextmanager
    def use_mem_pool(self, pool, device=None):
        self.events.append(("pool_enter", device))
        try:
            yield pool
        finally:
            self.events.append(("pool_exit", device))


class _FakeTorch:
    float16 = "torch.float16"
    float32 = "torch.float32"
    bfloat16 = "torch.bfloat16"
    int32 = "torch.int32"
    int64 = "torch.int64"
    contiguous_format = "torch.contiguous_format"

    def __init__(self, events, *, fail_linear=False):
        self.events = events
        self.cuda = _FakeCuda(events)
        self._C = SimpleNamespace(
            _cuda_clearCublasWorkspaces=lambda: events.append(("clear_cublas",))
        )
        self._next_output = 900000

        def wrap(session, lease, allocation, offset, sizes, strides, scalar_type):
            event = (
                "wrap",
                int(session),
                int(lease),
                int(allocation),
                int(offset),
                tuple(sizes),
                tuple(strides),
                int(scalar_type),
            )
            events.append(event)
            return _ManagedView("{}:{}".format(allocation, offset))

        aten = SimpleNamespace()
        for name in (
            "embedding",
            "linear",
            "mm",
            "addmm",
            "bmm",
            "add",
            "mul",
            "neg",
            "cat",
            "silu",
            "clone",
            "_to_copy",
            "view_copy",
            "copy",
            "rms_norm",
            "scaled_dot_product_attention",
        ):
            setattr(
                aten,
                name,
                SimpleNamespace(
                    out=_Op(events, name + ".out", fail=fail_linear and name == "linear"),
                    default=_Op(events, name + ".default"),
                ),
            )
        self.ops = SimpleNamespace(
            aten=aten,
            xvram_internal=SimpleNamespace(_wrap_resolved_v1=wrap),
        )

        @contextmanager
        def sdpa_kernel(*, backends):
            events.append(("sdpa_enter", tuple(backends)))
            try:
                yield
            finally:
                events.append(("sdpa_exit",))

        self.nn = SimpleNamespace(
            attention=SimpleNamespace(
                sdpa_kernel=sdpa_kernel,
                SDPBackend=SimpleNamespace(MATH="math", FLASH_ATTENTION="flash"),
            )
        )

    def empty(self, count, device=None):
        self.events.append(("empty", int(count), device))
        return object()

    def empty_strided(self, sizes, strides, dtype=None, device=None):
        self.events.append(
            ("empty_strided", tuple(sizes), tuple(strides), dtype, device)
        )
        self._next_output += 4096
        return _HostOutput(sizes, strides, dtype, self._next_output)


class _FakeControl:
    def __init__(self, events, *, target=1 << 30):
        self.events = events
        self.target = target
        self.next_allocation = 100
        self.next_lease = 1000
        self.allocations = {}
        self.sealed = {}

    def create_session(self, config):
        self.events.append(("session_create", config.device, config.scratch_bytes))
        return 7

    def create_allocation(self, session, bytes_value, hint):
        handle = self.next_allocation
        self.next_allocation += 1
        self.allocations[handle] = int(bytes_value)
        self.events.append(("allocation_create", session, handle, bytes_value, hint))
        return handle

    def write(self, allocation, offset, pointer, length):
        self.events.append(("write", allocation, offset, pointer, length))

    def read(self, allocation, offset, pointer, length):
        self.events.append(("read", allocation, offset, pointer, length))

    def prefetch(self, session, ranges):
        self.events.append(("prefetch", session, tuple(ranges)))

    def acquire(self, session, ranges):
        lease = self.next_lease
        self.next_lease += 1
        self.events.append(("acquire", session, lease, tuple(ranges)))
        return NativeLeaseInfo(lease, 1, 0, 0, len(ranges), 0.0, 0xABC0 + lease)

    def lease_info(self, lease):
        self.events.append(("lease_info", lease))
        return NativeLeaseInfo(lease, 1, 0, 0, 3, 0.0, 0xABC0 + lease)

    def seal(self, lease, mode):
        self.sealed[lease] = mode
        self.events.append(("seal", lease, mode))

    def wait(self, lease, timeout):
        self.events.append(("wait", lease, timeout))
        mode = self.sealed.get(lease)
        if mode == 1:
            return NativeLeaseInfo(lease, 3, 0, 0, 3, 1.5, 0)
        if mode == 2:
            return NativeLeaseInfo(lease, 4, 0, 0, 3, 0.0, 0)
        return NativeLeaseInfo(lease, 5, 12, 0, 3, 0.0, 0)

    def discard(self, allocation, offset, length):
        self.events.append(("discard", allocation, offset, length))

    def release(self, allocation):
        self.events.append(("release", allocation))
        self.allocations.pop(allocation, None)

    def close_session(self, session, timeout):
        self.events.append(("session_close", session, timeout))

    def telemetry(self, session):
        self.events.append(("telemetry", session))
        return {
            "cache_target_bytes": self.target,
            "stable_addresses": True,
            "no_physical_aliases": True,
            "tensor_views_live": 0,
        }


class _FakeGemmControl(_FakeControl):
    def execute_gemm(self, session, problem):
        self.events.append(("gemm_execute", session, problem))
        return {
            "status": 1,
            "boundary": 5,
            "native_code": 0,
            "tile_count": 3,
            "tiles_submitted": 3,
            "tiles_completed": 3,
            "events_recorded": 3,
            "events_retired": 3,
            "maps": 5,
            "set_access_calls": 5,
            "h2d_bytes": 128,
            "d2h_bytes": 0,
            "clean_evictions": 1,
            "dirty_evictions": 0,
            "handles_reused": 2,
            "prefetches": 1,
            "cublas_core_tiles": 3,
            "cublas_lt_tiles": 0,
            "algorithm_cache_hits": 0,
            "workspace_bytes": 4 << 20,
            "maximum_working_set_bytes": 64 << 20,
            "tile_m": 128,
            "tile_n": 64,
            "tile_k": 32,
            "maximum_kernel_milliseconds": 1.25,
            "elapsed_milliseconds": 4.0,
        }


class TorchNativeTests(unittest.TestCase):
    def _fixture(self, *, target=1 << 30, fail_linear=False):
        exported = _embedding_linear_program()
        provider = _Provider(exported)
        plan = build_inference_plan(exported, state_provider=provider)
        events = []
        torch_module = _FakeTorch(events, fail_linear=fail_linear)
        control = _FakeControl(events, target=target)
        backend = NativeInferenceBackend(
            config=NativeRuntimeConfig(),
            _torch_module=torch_module,
            _control=control,
        )
        return plan, provider, backend, control, events

    def test_ctypes_v1_layout_is_size_tagged_and_telemetry_complete(self):
        self.assertEqual(_SessionConfigV1.struct_size.offset, 0)
        self.assertEqual(_SessionConfigV1.abi_version.offset, 4)
        self.assertGreaterEqual(_ApiV1.reserved.offset, 2 * 4 + 17 * 8)
        self.assertGreater(_TelemetryV1.scratch_bytes_peak.offset, 0)

    def test_config_uses_phase4b_defaults_and_validates(self):
        config = NativeRuntimeConfig().validate()
        self.assertEqual(config.chunk_bytes, 64 << 20)
        self.assertEqual(config.cache_target_bytes, 0)
        self.assertEqual(config.headroom_bytes, 512 << 20)
        self.assertEqual(config.scratch_bytes, 512 << 20)
        with self.assertRaisesRegex(ValueError, "policy"):
            NativeRuntimeConfig(policy="fifo").validate()
        with self.assertRaisesRegex(ValueError, "staging"):
            NativeRuntimeConfig(staging_slots=1).validate()

    def test_prepare_initializes_primary_context_before_native_session(self):
        plan, provider, backend, control, events = self._fixture()
        backend.prepare(plan, provider)

        names = [item[0] for item in events]
        self.assertLess(names.index("set_device"), names.index("session_create"))
        self.assertLess(names.index("empty"), names.index("session_create"))
        # Two persistent storages are written once; activation/input storages
        # are allocated without materializing CUDA tensors.
        writes = [item for item in events if item[0] == "write"]
        self.assertEqual(len(writes), 2)
        self.assertGreaterEqual(len(control.allocations), 5)
        self.assertFalse(any(item[0] == "external_stream" for item in events))

        backend.close()
        self.assertFalse(control.allocations)

    def test_chunked_tied_state_streams_one_complete_storage_with_offsets(self):
        provider = _ChunkedProvider()
        plan = build_inference_plan(_chunked_tied_program(), state_provider=provider)
        self.assertEqual(plan.value("p_a").storage_id, plan.value("p_b").storage_id)
        events = []
        control = _FakeControl(events)
        backend = NativeInferenceBackend(
            config=NativeRuntimeConfig(chunk_size=16),
            _torch_module=_FakeTorch(events),
            _control=control,
        )

        backend.prepare(plan, provider)

        shared = backend._allocations[plan.value("p_a").storage_id]
        writes = [item for item in events if item[0] == "write" and item[1] == shared]
        self.assertEqual(
            [(item[2], item[4]) for item in writes],
            [(0, 16), (16, 16), (32, 16), (48, 16)],
        )
        self.assertEqual(sum(item[4] for item in writes), 64)
        self.assertEqual(len({item[1] for item in writes}), 1)
        backend.close()

    def test_scheduler_executes_exact_out_ops_on_runtime_external_stream(self):
        plan, provider, backend, _control, events = self._fixture()
        runtime = InferenceRuntime(plan, backend, provider)
        tokens = _Tensor(
            (4,), dtype="torch.int64", element_size=8, pointer=7000, data=[1, 2, 3, 5]
        )

        output = runtime.run(tokens)

        self.assertEqual(output.size(), (4, 3))
        op_names = [item[1] for item in events if item[0] == "op"]
        self.assertEqual(op_names, ["embedding.out", "linear.out"])
        stream_enters = [item for item in events if item[0] == "stream_enter"]
        self.assertEqual(len(stream_enters), 2)
        wraps = [item for item in events if item[0] == "wrap"]
        self.assertTrue(wraps)
        # The embedding weight is wrapped with full metadata even though only
        # selected row ranges participate in the lease.
        self.assertIn((8, 4), [item[5] for item in wraps])

        for seal in [index for index, item in enumerate(events) if item[0] == "seal"]:
            preceding_info = max(
                index for index, item in enumerate(events[:seal]) if item[0] == "lease_info"
            )
            self.assertLess(preceding_info, seal)
        waits = [index for index, item in enumerate(events) if item[0] == "wait"]
        discards = [index for index, item in enumerate(events) if item[0] == "discard"]
        self.assertTrue(discards)
        self.assertLess(waits[0], discards[0])

        telemetry = backend.telemetry_snapshot()
        self.assertEqual(telemetry["nodes_submitted"], 2)
        self.assertEqual(telemetry["nodes_retired"], 2)
        self.assertTrue(telemetry["stable_addresses"])
        runtime.close()

    def test_failed_aten_submission_uses_failed_after_submission_seal(self):
        plan, provider, backend, control, events = self._fixture(fail_linear=True)
        runtime = InferenceRuntime(plan, backend, provider)
        tokens = _Tensor(
            (4,), dtype="torch.int64", element_size=8, pointer=7000, data=[1, 2, 3, 5]
        )
        with self.assertRaisesRegex(RuntimeError, "injected ATen"):
            runtime.run(tokens)
        modes = [item[2] for item in events if item[0] == "seal"]
        self.assertEqual(modes[-1], 3)
        self.assertTrue(any(item[0] == "wait" for item in events))
        with self.assertRaisesRegex(BackendContractError, "stale"):
            # Failed submissions never escape as scheduler generation tokens.
            backend.retire(next(iter(control.sealed), None))
        runtime.close()

    def test_oversized_linear_never_attempts_a_full_working_set_lease(self):
        plan, provider, backend, _control, events = self._fixture(target=1)
        backend.prepare(plan, provider)
        linear = next(node for node in plan.nodes if node.adapter == "linear_out")
        request = ExecutionRequest(1, linear, linear.accesses)

        with self.assertRaisesRegex(BackendContractError, "gemm_execute"):
            backend.submit(request)

        self.assertFalse(any(item[0] == "acquire" for item in events))
        backend.close()

    def test_oversized_linear_routes_to_synchronous_native_tiled_gemm(self):
        exported = _embedding_linear_program()
        provider = _Provider(exported)
        plan = build_inference_plan(exported, state_provider=provider)
        events = []
        control = _FakeGemmControl(events, target=1)
        backend = NativeInferenceBackend(
            config=NativeRuntimeConfig(prefetch_distance=2),
            _torch_module=_FakeTorch(events),
            _control=control,
        )
        backend.prepare(plan, provider)
        linear = next(node for node in plan.nodes if node.adapter == "linear_out")

        token = backend.submit(ExecutionRequest(9, linear, linear.accesses))
        backend.retire(token)

        self.assertFalse(any(item[0] == "acquire" for item in events))
        _name, session, problem = next(item for item in events if item[0] == "gemm_execute")
        self.assertEqual(session, backend.session_handle)
        self.assertEqual((problem.m, problem.n, problem.k), (4, 3, 4))
        self.assertEqual(problem.b.operation, 2)
        self.assertEqual(problem.prefetch_distance, 2)
        telemetry = backend.telemetry_snapshot()
        self.assertEqual(telemetry["gemm_tiles_submitted"], 3)
        self.assertEqual(telemetry["gemm_events_retired"], 3)
        self.assertEqual(telemetry["nodes_retired"], 1)
        backend.close()

    def test_prefetch_and_dead_discard_use_opaque_allocation_handles(self):
        plan, provider, backend, _control, events = self._fixture()
        backend.prepare(plan, provider)
        embedding = next(node for node in plan.nodes if node.adapter == "embedding_out")
        backend.prefetch((embedding.accesses[0],))
        event = next(item for item in events if item[0] == "prefetch")
        self.assertIsInstance(event[2][0][0], int)
        self.assertNotEqual(event[2][0][0], 0)

        backend.release_values((embedding.output_value,))
        discard = next(item for item in events if item[0] == "discard")
        self.assertEqual(discard[2], 0)
        self.assertGreater(discard[3], 0)
        backend.close()

    def test_scratch_adapter_uses_bounded_pool_and_explicit_sdpa_backend(self):
        _plan, _provider, backend, _control, events = self._fixture()
        backend._scratch_allocator = _ScratchAllocator()
        output = _ManagedView("managed-output")

        backend._dispatch_scratch(
            "sdpa_math_scratch_copy",
            (_ManagedView("q"), _ManagedView("k"), _ManagedView("v")),
            {"is_causal": True},
            output,
        )

        names = [item[0] for item in events]
        self.assertIn("pool_enter", names)
        self.assertIn("sdpa_enter", names)
        self.assertIn(("op", "copy.out", mock.ANY, mock.ANY), events)
        self.assertLess(names.index("pool_exit"), names.index("pool_destroy"))

    def test_find_library_prefers_explicit_then_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xvram_torch_runtime.dll"
            path.write_bytes(b"fixture")
            self.assertEqual(find_runtime_library(path), path.resolve())
            with mock.patch.dict(os.environ, {"XVRAM_TORCH_RUNTIME_LIBRARY": str(path)}):
                self.assertEqual(find_runtime_library(), path.resolve())
        with self.assertRaisesRegex(NativeTorchRuntimeError, "does not exist"):
            find_runtime_library("definitely-missing-xvram-runtime.dll")


if __name__ == "__main__":
    unittest.main()
