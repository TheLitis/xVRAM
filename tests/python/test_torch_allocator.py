from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import xvram.torch_allocator as adapter  # noqa: E402


class _FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeNative:
    def __init__(self):
        self.reset_calls = 0
        self.get_status = 0
        self.reset_status = 0
        self.xvram_torch_get_stats = _FakeFunction(self._get_stats)
        self.xvram_torch_reset_stats = _FakeFunction(self._reset_stats)
        self.xvram_torch_get_last_error = _FakeFunction(self._get_last_error)

    def _get_stats(self, pointer, size):
        stats = ctypes.cast(pointer, ctypes.POINTER(adapter._NativeStatsV1)).contents
        stats.struct_size = int(size)
        stats.abi_version = adapter.TORCH_ALLOCATOR_ABI_VERSION
        stats.allocation_calls = 7
        stats.free_calls = 5
        stats.requested_bytes_current = 4096
        stats.mapped_bytes_peak = 8192
        stats.active_segments = 2
        stats.handles_created = 3
        stats.set_access_calls = 3
        stats.event_boundaries = 5
        stats.unsafe_unmaps = 0
        stats.last_native_error = -17
        stats.last_status = 4
        return self.get_status

    def _reset_stats(self):
        self.reset_calls += 1
        return self.reset_status

    @staticmethod
    def _get_last_error(pointer, _size):
        error = ctypes.cast(pointer, ctypes.POINTER(adapter._NativeErrorV1)).contents
        error.status = 4
        error.native_code = -17
        error.stage = b"allocation"
        error.operation = b"cuMemCreate"
        error.message = b"injected failure"
        return 0


class _FakeUsePool:
    def __init__(self, pool):
        self.pool = pool
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self.pool

    def __exit__(self, *_args):
        self.exited = True
        return False


class _FailingExitPool(_FakeUsePool):
    def __init__(self, pool):
        super().__init__(pool)
        self.fail_once = True

    def __exit__(self, *_args):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("injected context exit failure")
        return super().__exit__(*_args)


class _FakeCudaAllocator:
    def __init__(self, path, alloc_name, free_name):
        self.path = path
        self.alloc_name = alloc_name
        self.free_name = free_name
        self.handle = object()

    def allocator(self):
        return self.handle


class _FakeMemPool:
    def __init__(self, *, allocator):
        self.allocator = allocator


class _FakeCuda:
    def __init__(self):
        self.contexts = []
        self.memory = SimpleNamespace(CUDAPluggableAllocator=_FakeCudaAllocator)
        self.MemPool = _FakeMemPool

    def use_mem_pool(self, pool):
        context = _FakeUsePool(pool)
        self.contexts.append(context)
        return context


def _fake_torch(path):
    return SimpleNamespace(__file__=str(path / "torch" / "__init__.py"), cuda=_FakeCuda())


class TorchAllocatorTests(unittest.TestCase):
    def _load(self, directory):
        library = directory / ("xvram_torch_allocator.dll" if os.name == "nt" else "libxvram_torch_allocator.so")
        library.touch()
        native = _FakeNative()
        torch_module = _fake_torch(directory)
        with mock.patch.object(adapter.ctypes, "CDLL", return_value=native), mock.patch.object(
            adapter, "_open_windows_dll_directories", return_value=()
        ):
            loaded = adapter.XvramTorchAllocator(library, _torch_module=torch_module)
        return loaded, native, library.resolve()

    def test_loader_passes_absolute_path_and_exact_callback_names(self):
        with tempfile.TemporaryDirectory() as value:
            loaded, _native, expected_path = self._load(Path(value))

            self.assertEqual(loaded.library_path, expected_path)
            self.assertEqual(loaded.torch_allocator.path, str(expected_path))
            self.assertEqual(loaded.torch_allocator.alloc_name, "xvram_torch_alloc")
            self.assertEqual(loaded.torch_allocator.free_name, "xvram_torch_free")

    def test_mem_pool_retains_owner_and_is_reentrant(self):
        with tempfile.TemporaryDirectory() as value:
            loaded, _native, _path = self._load(Path(value))
            pool = loaded.create_mem_pool()

            self.assertIs(pool.allocator, loaded)
            self.assertIs(pool.raw_pool.allocator, loaded.torch_allocator.handle)
            with pool:
                with pool:
                    self.assertTrue(loaded._torch.cuda.contexts[-1].entered)
            self.assertTrue(all(context.exited for context in loaded._torch.cuda.contexts))

    def test_mem_pool_retains_exit_token_until_cleanup_succeeds(self):
        with tempfile.TemporaryDirectory() as value:
            loaded, _native, _path = self._load(Path(value))
            pool = loaded.create_mem_pool()
            context = _FailingExitPool(pool.raw_pool)
            with mock.patch.object(pool, "use", return_value=context):
                pool.__enter__()
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    pool.__exit__(None, None, None)
                self.assertEqual(len(pool._contexts.stack), 1)
                self.assertFalse(pool.__exit__(None, None, None))
                self.assertEqual(pool._contexts.stack, [])

    def test_stats_snapshot_and_reset(self):
        with tempfile.TemporaryDirectory() as value:
            loaded, native, _path = self._load(Path(value))

            stats = loaded.get_stats()
            self.assertEqual(stats.abi_version, adapter.TORCH_ALLOCATOR_ABI_VERSION)
            self.assertEqual(stats.allocation_calls, 7)
            self.assertEqual(stats.free_calls, 5)
            self.assertEqual(stats.requested_bytes_current, 4096)
            self.assertEqual(stats.mapped_bytes_peak, 8192)
            self.assertEqual(stats.handles_created, 3)
            self.assertEqual(stats.last_native_error, -17)
            self.assertEqual(stats.to_dict()["unsafe_unmaps"], 0)

            error = loaded.get_last_error()
            self.assertEqual(error.status, 4)
            self.assertEqual(error.native_code, -17)
            self.assertEqual(error.stage, "allocation")
            self.assertEqual(error.operation, "cuMemCreate")
            self.assertEqual(error.message, "injected failure")

            loaded.reset_stats()
            self.assertEqual(native.reset_calls, 1)

    def test_native_status_errors_are_not_silenced(self):
        with tempfile.TemporaryDirectory() as value:
            loaded, native, _path = self._load(Path(value))
            native.get_status = 27
            with self.assertRaises(adapter.NativeAllocatorError) as get_error:
                loaded.get_stats()
            self.assertEqual(get_error.exception.status, 27)

            native.reset_status = 23
            with self.assertRaises(adapter.NativeAllocatorError) as reset_error:
                loaded.reset_stats()
            self.assertEqual(reset_error.exception.status, 23)

    def test_missing_library_and_environment_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            adapter.XvramTorchAllocator("missing-xvram-plugin.dll", _torch_module=_fake_torch(Path.cwd()))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, adapter.XVRAM_TORCH_ALLOCATOR_ENV):
                adapter.load()


if __name__ == "__main__":
    unittest.main()
