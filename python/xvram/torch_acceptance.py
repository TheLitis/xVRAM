"""Phase 4b PyTorch acceptance state and reference utilities.

The production inference path must not construct an ordinary CPU copy of the
complete model.  :class:`DeterministicLlamaStateProvider` therefore describes
meta tensors to the planner and regenerates requested byte ranges directly
from their absolute deterministic offsets.  A caller may keep one returned
chunk until it has copied it into xVRAM pageable backing; the provider itself
does not retain generated tensor data.

PyTorch is intentionally imported only inside functions and constructors.  In
particular, importing this module remains safe in the controller/no-PyTorch
environment used by the protocol and report tests.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .torch_planner import (
    CapturedInference,
    PlanError,
    StateTensorInfo,
    TensorSpec,
    capture_inference,
)
from .torch_runtime import StateProvider
from .torch_workload import Llama2LikeConfig, build_llama2_like, deterministic_values


DEFAULT_TORCH_SEED = 0x585652414D503034
DEFAULT_STATE_CHUNK_BYTES = 64 << 20
_GENERATION_BLOCK_ELEMENTS = 1 << 20
_STRUCTURED_PERIOD_ELEMENTS = 4093
_STATE_PATTERNS = frozenset({"incompressible", "structured"})


def structured_values(
    element_offset: int,
    element_count: int,
    *,
    seed: int,
    dtype: Any,
    scale: float = 0.02,
) -> Any:
    """Generate a losslessly compressible, numerically non-trivial state range.

    The same LCG used by the incompressible fixture is evaluated over a small
    prime-period address space.  A 64 KiB LZ4 block therefore contains several
    byte-identical periods while adjacent matrix rows begin at different
    phases.  Absolute offsets keep partial/chunked reads order-independent.
    """

    if element_offset < 0 or element_count < 0:
        raise ValueError("element offset and count must be non-negative")
    torch = _torch()
    indices = torch.arange(
        element_offset,
        element_offset + element_count,
        dtype=torch.int64,
        device="cpu",
    ).remainder(_STRUCTURED_PERIOD_ELEMENTS)
    mixed = (indices * 1103515245 + (int(seed) & 0x7FFFFFFF)) & 0x7FFFFFFF
    values = mixed.to(torch.float32).mul_(1.0 / 1073741824.0).sub_(1.0)
    return values.mul_(float(scale)).to(dtype=dtype)


@dataclass(frozen=True)
class StateChunk:
    """One target-relative, contiguous CPU byte range.

    ``data`` is a one-dimensional ``torch.uint8`` CPU tensor.  Keeping the
    offset target-relative lets a backend place it at
    ``StateTensorInfo.offset_bytes + offset_bytes`` in a shared/tied storage.
    """

    target: str
    offset_bytes: int
    data: Any

    @property
    def length_bytes(self) -> int:
        return int(self.data.numel())


class ChunkedStateTensor:
    """Lightweight metadata and byte reader carried by ``StateTensorInfo``.

    This object deliberately has no ``data_ptr`` method: it is not a persistent
    materialized tensor and must never be handed to ATen.  Backends should use
    :meth:`read_bytes` or :meth:`iter_chunks` while filling xVRAM-owned pageable
    storage.
    """

    def __init__(
        self,
        *,
        target: str,
        dtype: Any,
        spec: TensorSpec,
        storage_identity: object,
        storage: "_GeneratedStorage",
        target_offset_bytes: int,
        default_chunk_bytes: int,
    ) -> None:
        self.target = target
        self.dtype = dtype
        self.device = "cpu-streamed"
        self.spec = spec
        self.storage_identity = storage_identity
        self._storage = storage
        self._target_offset_bytes = target_offset_bytes
        self._default_chunk_bytes = default_chunk_bytes
        self._logical_bytes = spec.numel * spec.element_size

    def size(self) -> Tuple[int, ...]:
        return self.spec.sizes

    def stride(self) -> Tuple[int, ...]:
        return self.spec.strides

    def element_size(self) -> int:
        return self.spec.element_size

    def storage_offset(self) -> int:
        return self.spec.storage_offset_elements

    def nbytes(self) -> int:
        return self._logical_bytes

    def read_bytes(self, offset_bytes: int, length_bytes: int) -> Any:
        """Generate one target-relative byte range without caching it."""

        _validate_byte_range(offset_bytes, length_bytes, self._logical_bytes)
        return self._storage.read_bytes(
            self._target_offset_bytes + offset_bytes, length_bytes
        )

    def iter_chunks(self, chunk_bytes: Optional[int] = None) -> Iterator[StateChunk]:
        size = self._default_chunk_bytes if chunk_bytes is None else int(chunk_bytes)
        if size <= 0:
            raise ValueError("chunk_bytes must be positive")
        offset = 0
        while offset < self._logical_bytes:
            length = min(size, self._logical_bytes - offset)
            yield StateChunk(self.target, offset, self.read_bytes(offset, length))
            offset += length


class _GeneratedStorage:
    def __init__(self, *, dtype: Any, storage_bytes: int) -> None:
        if storage_bytes <= 0:
            raise ValueError("generated state storage must be non-empty")
        self.dtype = dtype
        self.storage_bytes = int(storage_bytes)
        self.element_size = _dtype_element_size(dtype)
        if self.storage_bytes % self.element_size:
            raise ValueError("state storage size is not element aligned")

    def read_bytes(self, offset_bytes: int, length_bytes: int) -> Any:
        _validate_byte_range(offset_bytes, length_bytes, self.storage_bytes)
        torch = _torch()
        first = offset_bytes // self.element_size
        prefix = offset_bytes % self.element_size
        covered = prefix + length_bytes
        count = (covered + self.element_size - 1) // self.element_size
        # ``deterministic_values`` uses int64 mixing temporaries.  Filling a
        # 64 MiB FP16 chunk in bounded blocks avoids a several-hundred-MiB
        # transient while retaining one exact output chunk.
        values = torch.empty(count, dtype=self.dtype, device="cpu")
        generated = 0
        while generated < count:
            block = min(_GENERATION_BLOCK_ELEMENTS, count - generated)
            values[generated : generated + block].copy_(
                self._generate_elements(first + generated, block)
            )
            generated += block
        raw = values.view(torch.uint8).reshape(-1)
        return raw[prefix : prefix + length_bytes]

    def _generate_elements(self, element_offset: int, element_count: int) -> Any:
        raise NotImplementedError


class _ParameterStorage(_GeneratedStorage):
    def __init__(
        self,
        *,
        dtype: Any,
        storage_bytes: int,
        absolute_element_offset: int,
        seed: int,
        scale: float,
        state_pattern: str,
    ) -> None:
        super().__init__(dtype=dtype, storage_bytes=storage_bytes)
        self._absolute_element_offset = int(absolute_element_offset)
        self._seed = int(seed)
        self._scale = float(scale)
        self._state_pattern = _normalize_state_pattern(state_pattern)

    def _generate_elements(self, element_offset: int, element_count: int) -> Any:
        absolute_offset = self._absolute_element_offset + element_offset
        if self._state_pattern == "structured":
            return structured_values(
                absolute_offset,
                element_count,
                seed=self._seed,
                dtype=self.dtype,
                scale=self._scale,
            )
        return deterministic_values(
            absolute_offset,
            element_count,
            seed=self._seed,
            dtype=self.dtype,
            scale=self._scale,
        )


class _RopeStorage(_GeneratedStorage):
    def __init__(
        self,
        *,
        dtype: Any,
        storage_bytes: int,
        head_dim: int,
        function: str,
    ) -> None:
        super().__init__(dtype=dtype, storage_bytes=storage_bytes)
        if function not in {"cosine", "sine"}:
            raise ValueError("unsupported RoPE function")
        self._head_dim = int(head_dim)
        self._half_dim = self._head_dim // 2
        self._function = function

    def _generate_elements(self, element_offset: int, element_count: int) -> Any:
        torch = _torch()
        indices = torch.arange(
            element_offset,
            element_offset + element_count,
            dtype=torch.int64,
            device="cpu",
        )
        positions = torch.div(indices, self._half_dim, rounding_mode="floor").to(
            torch.float32
        )
        frequency_indices = (indices.remainder(self._half_dim) * 2).to(torch.float32)
        base = torch.tensor(10000.0, dtype=torch.float32)
        frequencies = torch.pow(base, -frequency_indices / float(self._head_dim))
        angles = positions * frequencies
        values = angles.cos() if self._function == "cosine" else angles.sin()
        return values.to(dtype=self.dtype)


class DeterministicLlamaStateProvider(StateProvider):
    """Chunk-streaming state for a strict meta Llama-2-like module.

    Parameter bytes use :func:`deterministic_values` over one monotonically
    increasing element address space.  Tied tensors share one generated
    storage and one ``storage_key``.  RoPE buffers are generated analytically
    with the exact float32 construction used by :func:`build_llama2_like`.
    """

    def __init__(
        self,
        module: Any,
        config: Llama2LikeConfig,
        *,
        seed: int = DEFAULT_TORCH_SEED,
        dtype: Any = None,
        chunk_bytes: int = DEFAULT_STATE_CHUNK_BYTES,
        scale: float = 0.02,
        state_pattern: str = "incompressible",
    ) -> None:
        torch = _torch()
        if getattr(module, "training", False):
            raise PlanError("streaming acceptance state requires module.eval()")
        self.config = config
        self.seed = int(seed)
        self.dtype = _resolve_dtype(dtype, torch)
        self.chunk_bytes = int(chunk_bytes)
        self.scale = float(scale)
        self.state_pattern = _normalize_state_pattern(state_pattern)
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and positive")

        state = _collect_meta_state(module)
        expected = _expected_state_shapes(config)
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            extra = sorted(set(state) - set(expected))
            raise PlanError(
                "meta Llama state targets differ from the acceptance contract: "
                "missing={!r}, extra={!r}".format(missing, extra)
            )

        self._infos: Dict[str, StateTensorInfo] = {}
        self._sources: Dict[str, ChunkedStateTensor] = {}
        generated: Dict[Tuple[str, int], Tuple[object, _GeneratedStorage, str]] = {}
        parameter_cursor = 0
        parameter_bytes = 0
        buffer_bytes = 0
        next_storage = 0

        for target, (tensor, state_kind) in state.items():
            spec = _meta_tensor_spec(target, tensor)
            if spec.sizes != expected[target]:
                raise PlanError(
                    "state target {!r} has shape {}, expected {}".format(
                        target, spec.sizes, expected[target]
                    )
                )
            if tensor.dtype != self.dtype:
                raise PlanError(
                    "state target {!r} has dtype {}, expected {}".format(
                        target, tensor.dtype, self.dtype
                    )
                )
            if not bool(tensor.is_contiguous()):
                raise PlanError("acceptance state tensors must be contiguous", node=target)
            storage = tensor.untyped_storage()
            storage_bytes = int(storage.nbytes())
            raw_storage_key = ("meta", int(getattr(storage, "_cdata", id(tensor))))
            existing = generated.get(raw_storage_key)
            if existing is None:
                storage_identity = ("xvram.llama_state", next_storage)
                next_storage += 1
                if state_kind == "parameter":
                    generated_storage: _GeneratedStorage = _ParameterStorage(
                        dtype=self.dtype,
                        storage_bytes=storage_bytes,
                        absolute_element_offset=parameter_cursor,
                        seed=self.seed,
                        scale=self.scale,
                        state_pattern=self.state_pattern,
                    )
                    parameter_cursor += storage_bytes // spec.element_size
                    parameter_bytes += storage_bytes
                elif target == "rope_cosine":
                    generated_storage = _RopeStorage(
                        dtype=self.dtype,
                        storage_bytes=storage_bytes,
                        head_dim=config.head_dim,
                        function="cosine",
                    )
                    buffer_bytes += storage_bytes
                elif target == "rope_sine":
                    generated_storage = _RopeStorage(
                        dtype=self.dtype,
                        storage_bytes=storage_bytes,
                        head_dim=config.head_dim,
                        function="sine",
                    )
                    buffer_bytes += storage_bytes
                else:
                    raise PlanError(
                        "unsupported acceptance buffer {!r}".format(target), node=target
                    )
                generated[raw_storage_key] = (
                    storage_identity,
                    generated_storage,
                    state_kind,
                )
            else:
                storage_identity, generated_storage, original_kind = existing
                if original_kind != state_kind:
                    raise PlanError("parameter/buffer storage aliasing is unsupported", node=target)
                if (
                    generated_storage.storage_bytes != storage_bytes
                    or generated_storage.dtype != self.dtype
                ):
                    raise PlanError("tied storage metadata is inconsistent", node=target)

            offset_bytes = spec.storage_offset_elements * spec.element_size
            if offset_bytes + spec.span_bytes > storage_bytes:
                raise PlanError("state tensor exceeds its meta storage", node=target)
            source = ChunkedStateTensor(
                target=target,
                dtype=self.dtype,
                spec=spec,
                storage_identity=storage_identity,
                storage=generated_storage,
                target_offset_bytes=offset_bytes,
                default_chunk_bytes=self.chunk_bytes,
            )
            self._sources[target] = source
            self._infos[target] = StateTensorInfo(
                target=target,
                tensor=source,
                storage_key=storage_identity,
                storage_bytes=storage_bytes,
                offset_bytes=offset_bytes,
                spec=spec,
            )

        self.parameter_bytes = parameter_bytes
        self.buffer_bytes = buffer_bytes
        self.logical_state_bytes = parameter_bytes + buffer_bytes

    def describe(self, target: str) -> StateTensorInfo:
        try:
            return self._infos[target]
        except KeyError as error:
            raise PlanError("module state target {!r} is unavailable".format(target)) from error

    def targets(self) -> Tuple[str, ...]:
        return tuple(sorted(self._infos))

    def source(self, target: str) -> ChunkedStateTensor:
        try:
            return self._sources[target]
        except KeyError as error:
            raise PlanError("module state target {!r} is unavailable".format(target)) from error

    def read_bytes(self, target: str, offset_bytes: int, length_bytes: int) -> Any:
        return self.source(target).read_bytes(offset_bytes, length_bytes)

    def iter_chunks(
        self, target: str, chunk_bytes: Optional[int] = None
    ) -> Iterator[StateChunk]:
        return self.source(target).iter_chunks(chunk_bytes)

    def materialize_tensor(self, target: str) -> Any:
        """Materialize one target for the streamed ordinary-PyTorch reference.

        The result is never cached.  At most this one tensor plus one generated
        chunk exists on the host during construction.
        """

        torch = _torch()
        info = self.describe(target)
        source = self.source(target)
        raw = torch.empty(source.nbytes(), dtype=torch.uint8, device="cpu")
        for chunk in source.iter_chunks():
            raw[chunk.offset_bytes : chunk.offset_bytes + chunk.length_bytes].copy_(
                chunk.data
            )
        typed = raw.view(self.dtype)
        if info.spec.storage_offset_elements != 0:
            # ``raw`` begins at the target rather than the complete shared
            # storage, so its local storage offset is always zero.
            storage_offset = 0
        else:
            storage_offset = info.spec.storage_offset_elements
        return torch.as_strided(
            typed,
            size=info.spec.sizes,
            stride=info.spec.strides,
            storage_offset=storage_offset,
        )


@dataclass(frozen=True)
class PreparedMetaLlama:
    config: Llama2LikeConfig
    module: Any
    state_provider: DeterministicLlamaStateProvider
    example_inputs: Tuple[Any, ...]
    dtype: Any
    seed: int


@dataclass(frozen=True)
class StrictMetaLlamaExport:
    prepared: PreparedMetaLlama
    captured: CapturedInference


def deterministic_token_ids(
    config: Llama2LikeConfig,
    *,
    seed: int = DEFAULT_TORCH_SEED,
) -> Any:
    """Return the canonical contiguous CPU ``int64`` acceptance input."""

    torch = _torch()
    count = config.batch * config.sequence
    indices = torch.arange(count, dtype=torch.int64, device="cpu")
    values = (indices * 1103515245 + (int(seed) & 0x7FFFFFFF)).remainder(
        config.vocab_size
    )
    return values.reshape(config.batch, config.sequence).contiguous()


def prepare_meta_llama(
    config: Llama2LikeConfig,
    *,
    seed: int = DEFAULT_TORCH_SEED,
    dtype: Any = None,
    chunk_bytes: int = DEFAULT_STATE_CHUNK_BYTES,
    state_pattern: str = "incompressible",
) -> PreparedMetaLlama:
    """Build a strict meta model, streaming provider, and CPU example input."""

    torch = _torch()
    resolved_dtype = _resolve_dtype(dtype, torch)
    module = build_llama2_like(config, device="meta", dtype=resolved_dtype).eval()
    provider = DeterministicLlamaStateProvider(
        module,
        config,
        seed=seed,
        dtype=resolved_dtype,
        chunk_bytes=chunk_bytes,
        state_pattern=state_pattern,
    )
    example = deterministic_token_ids(config, seed=seed)
    return PreparedMetaLlama(
        config=config,
        module=module,
        state_provider=provider,
        example_inputs=(example,),
        dtype=resolved_dtype,
        seed=int(seed),
    )


def strict_export_meta_llama(
    config: Llama2LikeConfig,
    *,
    seed: int = DEFAULT_TORCH_SEED,
    dtype: Any = None,
    chunk_bytes: int = DEFAULT_STATE_CHUNK_BYTES,
    state_pattern: str = "incompressible",
) -> StrictMetaLlamaExport:
    """Run canonical ``torch.export.export(..., strict=True)`` on the meta model."""

    torch = _torch()
    prepared = prepare_meta_llama(
        config,
        seed=seed,
        dtype=dtype,
        chunk_bytes=chunk_bytes,
        state_pattern=state_pattern,
    )
    captured = capture_inference(
        prepared.module,
        prepared.example_inputs,
        state_provider=prepared.state_provider,
        torch_module=torch,
    )
    return StrictMetaLlamaExport(prepared=prepared, captured=captured)


@dataclass(frozen=True)
class OutputError:
    element_count: int
    mismatch_count: int
    max_absolute: float
    max_relative: float
    mean_absolute: float
    atol: float
    rtol: float
    within_tolerance: bool


def output_digest(tensor: Any) -> str:
    """Hash tensor metadata and exact contiguous bytes with SHA-256."""

    torch = _torch()
    value = tensor.detach().to(device="cpu").contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": [int(item) for item in value.shape]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    raw = value.view(torch.uint8).reshape(-1).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def output_error(
    actual: Any,
    reference: Any,
    *,
    atol: float = 1.0e-2,
    rtol: float = 1.0e-2,
) -> OutputError:
    """Return strict elementwise error statistics on CPU float64 values."""

    torch = _torch()
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    left = actual.detach().to(device="cpu")
    right = reference.detach().to(device="cpu")
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("actual and reference shapes differ")
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    count = int(left64.numel())
    if count == 0:
        return OutputError(0, 0, 0.0, 0.0, 0.0, atol, rtol, True)
    finite = torch.isfinite(left64) & torch.isfinite(right64)
    difference = (left64 - right64).abs()
    tolerance = atol + rtol * right64.abs()
    close = finite & (difference <= tolerance)
    mismatches = int((~close).sum().item())
    if not bool(finite.all()):
        maximum_absolute = math.inf
        maximum_relative = math.inf
        mean_absolute = math.inf
    else:
        maximum_absolute = float(difference.max().item())
        mean_absolute = float(difference.mean().item())
        denominator = right64.abs().clamp_min(torch.finfo(torch.float64).tiny)
        maximum_relative = float((difference / denominator).max().item())
    return OutputError(
        element_count=count,
        mismatch_count=mismatches,
        max_absolute=maximum_absolute,
        max_relative=maximum_relative,
        mean_absolute=mean_absolute,
        atol=float(atol),
        rtol=float(rtol),
        within_tolerance=mismatches == 0,
    )


@dataclass(frozen=True)
class ReferenceProgress:
    stage: str
    completed: int
    total: int


@dataclass(frozen=True)
class StreamedReferenceResult:
    output: Any
    digest: str
    elapsed_ms: float
    layers_completed: int
    peak_decoder_weight_bytes: int
    peak_parameter_bytes: int
    device: str
    dtype: str
    sdpa_backend: str


def run_streamed_decoder_layer(
    hidden_states: Any,
    *,
    layer_index: int,
    config: Llama2LikeConfig,
    state_provider: StateProvider,
    cosine: Any,
    sine: Any,
    device: Any,
    dtype: Any,
    sdpa_backend: str = "math",
    release_cuda_cache: bool = True,
) -> Any:
    """Execute one decoder layer with only that layer's weights resident.

    All weights are discarded before this function returns.  CUDA execution is
    synchronized first so allocator reuse cannot overlap two layer generations.
    The function also supports ``device='cpu'`` for small no-GPU correctness
    tests.
    """

    torch = _torch()
    functional = torch.nn.functional
    target_device = torch.device(device)
    if layer_index < 0 or layer_index >= config.layers:
        raise IndexError("decoder layer index is out of range")
    prefix = "layers.{}.".format(layer_index)
    suffixes = (
        "attention_norm.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "ffn_norm.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )
    weights = {
        suffix: _state_to_device(
            state_provider, prefix + suffix, target_device, dtype
        )
        for suffix in suffixes
    }
    try:
        residual = hidden_states
        normalized = functional.rms_norm(
            hidden_states,
            (config.hidden,),
            weights["attention_norm.weight"],
            1.0e-5,
        )

        def project(suffix: str) -> Any:
            value = functional.linear(normalized, weights[suffix], None)
            value = value.view(
                config.batch, config.sequence, config.heads, config.head_dim
            )
            return value.transpose(1, 2)

        query = _apply_rope(project("q_proj.weight"), cosine, sine)
        key = _apply_rope(project("k_proj.weight"), cosine, sine)
        value = project("v_proj.weight")
        with _sdpa_context(torch, sdpa_backend, target_device):
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )
        attended = attended.transpose(1, 2).reshape(
            config.batch, config.sequence, config.hidden
        )
        hidden_states = residual + functional.linear(
            attended, weights["o_proj.weight"], None
        )
        normalized = functional.rms_norm(
            hidden_states,
            (config.hidden,),
            weights["ffn_norm.weight"],
            1.0e-5,
        )
        feed_forward = functional.silu(
            functional.linear(normalized, weights["gate_proj.weight"], None)
        ) * functional.linear(normalized, weights["up_proj.weight"], None)
        result = hidden_states + functional.linear(
            feed_forward, weights["down_proj.weight"], None
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        return result
    finally:
        weights.clear()
        if target_device.type == "cuda" and release_cuda_cache:
            torch.cuda.empty_cache()


def run_layer_streamed_reference(
    prepared: PreparedMetaLlama,
    *,
    token_ids: Any = None,
    device: Any = "cuda:0",
    sdpa_backend: str = "math",
    release_cuda_cache: bool = True,
    progress: Optional[Callable[[ReferenceProgress], None]] = None,
) -> StreamedReferenceResult:
    """Run the ordinary-PyTorch reference without materializing a full model."""

    torch = _torch()
    functional = torch.nn.functional
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the layer-streamed reference")
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("reference device must be CPU or CUDA")
    tokens = prepared.example_inputs[0] if token_ids is None else token_ids
    _validate_token_ids(tokens, prepared.config)
    provider = prepared.state_provider
    config = prepared.config
    dtype = prepared.dtype
    start = time.perf_counter()
    peak_decoder = max(
        _layer_weight_bytes(provider, layer) for layer in range(config.layers)
    )
    embedding_bytes = provider.describe("embedding.weight").spec.span_bytes
    head_bytes = provider.describe("lm_head.weight").spec.span_bytes
    peak_parameters = max(peak_decoder, embedding_bytes, head_bytes)

    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)

    with torch.inference_mode():
        embedding = _state_to_device(
            provider, "embedding.weight", target_device, dtype
        )
        hidden_states = functional.embedding(tokens.to(target_device), embedding)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        del embedding
        if target_device.type == "cuda" and release_cuda_cache:
            torch.cuda.empty_cache()

        cosine = _state_to_device(provider, "rope_cosine", target_device, dtype)
        sine = _state_to_device(provider, "rope_sine", target_device, dtype)
        _emit_progress(progress, "embedding", 1, 1)
        for layer in range(config.layers):
            hidden_states = run_streamed_decoder_layer(
                hidden_states,
                layer_index=layer,
                config=config,
                state_provider=provider,
                cosine=cosine,
                sine=sine,
                device=target_device,
                dtype=dtype,
                sdpa_backend=sdpa_backend,
                release_cuda_cache=release_cuda_cache,
            )
            _emit_progress(progress, "decoder", layer + 1, config.layers)

        final_norm = _state_to_device(
            provider, "final_norm.weight", target_device, dtype
        )
        hidden_states = functional.rms_norm(
            hidden_states, (config.hidden,), final_norm, 1.0e-5
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        del final_norm, cosine, sine
        if target_device.type == "cuda" and release_cuda_cache:
            torch.cuda.empty_cache()

        head = _state_to_device(provider, "lm_head.weight", target_device, dtype)
        output = functional.linear(hidden_states, head, None)
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        host_output = output.to(device="cpu").contiguous()
        del head, output, hidden_states
        if target_device.type == "cuda" and release_cuda_cache:
            torch.cuda.empty_cache()
        _emit_progress(progress, "output", 1, 1)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return StreamedReferenceResult(
        output=host_output,
        digest=output_digest(host_output),
        elapsed_ms=elapsed_ms,
        layers_completed=config.layers,
        peak_decoder_weight_bytes=peak_decoder,
        peak_parameter_bytes=peak_parameters,
        device=str(target_device),
        dtype=str(dtype),
        sdpa_backend=_normalize_sdpa_backend(sdpa_backend),
    )


def _state_to_device(
    provider: StateProvider, target: str, device: Any, dtype: Any
) -> Any:
    torch = _torch()
    materialize = getattr(provider, "materialize_tensor", None)
    if callable(materialize):
        host = materialize(target)
    else:
        host = provider.describe(target).tensor
        if not isinstance(host, torch.Tensor):
            raise TypeError(
                "state provider must expose materialize_tensor() or ordinary tensors"
            )
    if getattr(getattr(host, "device", None), "type", None) != "cpu":
        raise ValueError("reference state must materialize in CPU memory")
    if host.dtype != dtype:
        raise ValueError("reference state dtype differs from the prepared dtype")
    return host.to(device=device, dtype=dtype, non_blocking=False)


def _apply_rope(value: Any, cosine: Any, sine: Any) -> Any:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    first = even * cosine + -(odd * sine)
    second = even * sine + odd * cosine
    return _torch().cat((first.unsqueeze(-1), second.unsqueeze(-1)), dim=-1).reshape(
        value.shape
    )


def _sdpa_context(torch: Any, backend: str, device: Any) -> Any:
    normalized = _normalize_sdpa_backend(backend)
    if normalized == "flash_attention" and device.type != "cuda":
        raise ValueError("FLASH_ATTENTION requires a CUDA reference device")
    attention = getattr(torch.nn, "attention", None)
    kernel = getattr(attention, "sdpa_kernel", None)
    enum = getattr(attention, "SDPBackend", None)
    if kernel is None or enum is None:
        if normalized != "math":
            raise RuntimeError("this PyTorch build cannot select an SDPA backend")
        return contextlib.nullcontext()
    selected = enum.MATH if normalized == "math" else enum.FLASH_ATTENTION
    return kernel([selected])


def _normalize_sdpa_backend(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "math": "math",
        "flash": "flash_attention",
        "flash_attention": "flash_attention",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError("sdpa_backend must be math or flash_attention") from error


def _emit_progress(
    callback: Optional[Callable[[ReferenceProgress], None]],
    stage: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None:
        callback(ReferenceProgress(stage, completed, total))


def _layer_weight_bytes(provider: StateProvider, layer: int) -> int:
    prefix = "layers.{}.".format(layer)
    suffixes = (
        "attention_norm.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "ffn_norm.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )
    return sum(provider.describe(prefix + suffix).spec.span_bytes for suffix in suffixes)


def _validate_token_ids(tokens: Any, config: Llama2LikeConfig) -> None:
    torch = _torch()
    if not isinstance(tokens, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if getattr(tokens.device, "type", None) != "cpu":
        raise ValueError("token_ids must reside on CPU")
    if tokens.dtype != torch.int64:
        raise ValueError("token_ids must use torch.int64")
    if tuple(tokens.shape) != (config.batch, config.sequence):
        raise ValueError("token_ids do not match the static acceptance shape")
    if not bool(tokens.is_contiguous()):
        raise ValueError("token_ids must be contiguous")
    if int(tokens.min().item()) < 0 or int(tokens.max().item()) >= config.vocab_size:
        raise ValueError("token_ids contain an out-of-range vocabulary index")


def _collect_meta_state(module: Any) -> Mapping[str, Tuple[Any, str]]:
    result: Dict[str, Tuple[Any, str]] = {}
    for method_name, kind in (("named_parameters", "parameter"), ("named_buffers", "buffer")):
        method = getattr(module, method_name, None)
        if method is None or not callable(method):
            raise TypeError("module does not expose {}()".format(method_name))
        try:
            values = method(recurse=True, remove_duplicate=False)
        except TypeError:
            values = method(recurse=True)
        for target, tensor in values:
            if not isinstance(target, str) or not target:
                raise PlanError("meta state targets must be non-empty strings")
            if target in result and result[target][0] is not tensor:
                raise PlanError("duplicate meta state target {!r}".format(target))
            device_type = getattr(getattr(tensor, "device", None), "type", None)
            if device_type != "meta":
                raise PlanError(
                    "streaming acceptance provider requires a meta module", node=target
                )
            result[target] = (tensor, kind)
    return result


def _expected_state_shapes(config: Llama2LikeConfig) -> Mapping[str, Tuple[int, ...]]:
    result: Dict[str, Tuple[int, ...]] = {
        "embedding.weight": (config.vocab_size, config.hidden),
        "final_norm.weight": (config.hidden,),
        "lm_head.weight": (config.vocab_size, config.hidden),
        "rope_cosine": (1, 1, config.sequence, config.head_dim // 2),
        "rope_sine": (1, 1, config.sequence, config.head_dim // 2),
    }
    for layer in range(config.layers):
        prefix = "layers.{}.".format(layer)
        result[prefix + "attention_norm.weight"] = (config.hidden,)
        result[prefix + "ffn_norm.weight"] = (config.hidden,)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            result[prefix + name + ".weight"] = (config.hidden, config.hidden)
        for name in ("gate_proj", "up_proj"):
            result[prefix + name + ".weight"] = (
                config.intermediate,
                config.hidden,
            )
        result[prefix + "down_proj.weight"] = (
            config.hidden,
            config.intermediate,
        )
    return result


def _meta_tensor_spec(target: str, tensor: Any) -> TensorSpec:
    sizes = tuple(int(item) for item in tensor.size())
    strides = tuple(int(item) for item in tensor.stride())
    if len(sizes) != len(strides) or not sizes or any(size <= 0 for size in sizes):
        raise PlanError("acceptance state requires non-empty static tensors", node=target)
    if any(stride < 0 for stride in strides):
        raise PlanError("negative state strides are unsupported", node=target)
    element_size = int(tensor.element_size())
    storage_offset = int(tensor.storage_offset())
    maximum = sum((size - 1) * stride for size, stride in zip(sizes, strides))
    span_bytes = (maximum + 1) * element_size
    numel = math.prod(sizes)
    return TensorSpec(
        dtype=str(tensor.dtype),
        element_size=element_size,
        sizes=sizes,
        strides=strides,
        storage_offset_elements=storage_offset,
        span_bytes=span_bytes,
        numel=numel,
    )


def _validate_byte_range(offset_bytes: int, length_bytes: int, total_bytes: int) -> None:
    if isinstance(offset_bytes, bool) or isinstance(length_bytes, bool):
        raise TypeError("byte offsets and lengths must be integers")
    offset = int(offset_bytes)
    length = int(length_bytes)
    if offset < 0 or length <= 0 or offset > total_bytes or length > total_bytes - offset:
        raise ValueError("byte range exceeds the generated state target")


def _dtype_element_size(dtype: Any) -> int:
    torch = _torch()
    return int(torch.empty((), dtype=dtype, device="cpu").element_size())


def _resolve_dtype(dtype: Any, torch: Any) -> Any:
    if dtype is None:
        return torch.float16
    if isinstance(dtype, str):
        normalized = dtype.strip().lower().replace("torch.", "")
        aliases = {
            "float16": torch.float16,
            "half": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "float": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError("dtype must be float16, bfloat16, or float32") from error
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    return dtype


def _normalize_state_pattern(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _STATE_PATTERNS:
        raise ValueError("state_pattern must be incompressible or structured")
    return normalized


def _torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("PyTorch is required for acceptance execution") from error
    return torch


__all__ = [
    "ChunkedStateTensor",
    "DEFAULT_STATE_CHUNK_BYTES",
    "DEFAULT_TORCH_SEED",
    "DeterministicLlamaStateProvider",
    "OutputError",
    "PreparedMetaLlama",
    "ReferenceProgress",
    "StateChunk",
    "StreamedReferenceResult",
    "StrictMetaLlamaExport",
    "deterministic_token_ids",
    "output_digest",
    "output_error",
    "prepare_meta_llama",
    "run_layer_streamed_reference",
    "run_streamed_decoder_layer",
    "strict_export_meta_llama",
    "structured_values",
]
