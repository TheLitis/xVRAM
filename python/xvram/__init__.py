"""xVRAM Python integration.

PyTorch is imported lazily only when an allocator is loaded.
"""

from .fx import (
    FxNextUsePlan,
    FxReadHint,
    FxStepPlan,
    FxValueUse,
    analyze_next_uses,
)
from .lifetime import (
    TensorLifetime,
    TensorLifetimeHint,
    TensorRole,
    classify_module_tensors,
    classify_tensor,
)
from .torch_allocator import (
    TORCH_ALLOCATOR_ABI_VERSION,
    XVRAM_TORCH_ALLOCATOR_ENV,
    NativeAllocatorError,
    NativeAllocatorErrorInfo,
    NativeAllocatorStats,
    TorchUnavailableError,
    XvramMemPool,
    XvramTorchAllocator,
    XvramTorchError,
    create_mem_pool,
    load,
)

__all__ = [
    "FxNextUsePlan",
    "FxReadHint",
    "FxStepPlan",
    "FxValueUse",
    "NativeAllocatorError",
    "NativeAllocatorErrorInfo",
    "NativeAllocatorStats",
    "TORCH_ALLOCATOR_ABI_VERSION",
    "TensorLifetime",
    "TensorLifetimeHint",
    "TensorRole",
    "TorchUnavailableError",
    "XVRAM_TORCH_ALLOCATOR_ENV",
    "XvramMemPool",
    "XvramTorchAllocator",
    "XvramTorchError",
    "analyze_next_uses",
    "classify_module_tensors",
    "classify_tensor",
    "create_mem_pool",
    "load",
]
