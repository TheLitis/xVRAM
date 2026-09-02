"""Deterministic Llama-2-like acceptance workload for Phase 4b."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Llama2LikeConfig:
    layers: int = 31
    vocab_size: int = 32000
    hidden: int = 4096
    intermediate: int = 11008
    heads: int = 32
    batch: int = 1
    sequence: int = 32

    def __post_init__(self) -> None:
        if min(
            self.layers,
            self.vocab_size,
            self.hidden,
            self.intermediate,
            self.heads,
            self.batch,
            self.sequence,
        ) < 1:
            raise ValueError("all Llama-2-like dimensions must be positive")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if (self.hidden // self.heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def parameter_elements(self) -> int:
        per_layer = (
            4 * self.hidden * self.hidden
            + 3 * self.hidden * self.intermediate
            + 2 * self.hidden
        )
        global_parameters = 2 * self.vocab_size * self.hidden + self.hidden
        return self.layers * per_layer + global_parameters

    def parameter_bytes(self, element_size: int = 2) -> int:
        if element_size < 1:
            raise ValueError("element_size must be positive")
        return self.parameter_elements() * element_size


def deterministic_values(
    element_offset: int,
    element_count: int,
    *,
    seed: int,
    dtype: Any,
    scale: float = 0.02,
) -> Any:
    """Generate a chunk independently from its absolute element offset.

    The 31-bit arithmetic is exactly representable in int64 for the accepted
    model sizes and makes chunked/meta-model loading independent of chunk order.
    """

    if element_offset < 0 or element_count < 0:
        raise ValueError("element offset and count must be non-negative")
    import torch

    indices = torch.arange(
        element_offset, element_offset + element_count, dtype=torch.int64, device="cpu"
    )
    mixed = (indices * 1103515245 + (seed & 0x7FFFFFFF)) & 0x7FFFFFFF
    values = mixed.to(torch.float32).mul_(1.0 / 1073741824.0).sub_(1.0)
    return values.mul_(scale).to(dtype=dtype)


def build_llama2_like(config: Llama2LikeConfig, *, device: Any = "meta", dtype: Any = None) -> Any:
    """Build the exact acceptance graph without allocating its full state."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    target_dtype = dtype or torch.float16

    class RMSNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(config.hidden, device=device, dtype=target_dtype))

        def forward(self, value: Any) -> Any:
            normalized = functional.rms_norm(value, (config.hidden,), self.weight, 1.0e-5)
            return normalized

    def rope(value: Any, cosine: Any, sine: Any) -> Any:
        # value: [batch, heads, sequence, head_dim]
        even = value[..., 0::2]
        odd = value[..., 1::2]
        first = even * cosine + -(odd * sine)
        second = even * sine + odd * cosine
        return torch.cat((first.unsqueeze(-1), second.unsqueeze(-1)), dim=-1).reshape(value.shape)

    class DecoderBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = RMSNorm()
            self.ffn_norm = RMSNorm()
            self.q_proj = nn.Linear(config.hidden, config.hidden, bias=False, device=device, dtype=target_dtype)
            self.k_proj = nn.Linear(config.hidden, config.hidden, bias=False, device=device, dtype=target_dtype)
            self.v_proj = nn.Linear(config.hidden, config.hidden, bias=False, device=device, dtype=target_dtype)
            self.o_proj = nn.Linear(config.hidden, config.hidden, bias=False, device=device, dtype=target_dtype)
            self.gate_proj = nn.Linear(config.hidden, config.intermediate, bias=False, device=device, dtype=target_dtype)
            self.up_proj = nn.Linear(config.hidden, config.intermediate, bias=False, device=device, dtype=target_dtype)
            self.down_proj = nn.Linear(config.intermediate, config.hidden, bias=False, device=device, dtype=target_dtype)

        def forward(self, hidden_states: Any, cosine: Any, sine: Any) -> Any:
            residual = hidden_states
            normalized = self.attention_norm(hidden_states)

            def project(projection: Any) -> Any:
                value = projection(normalized)
                value = value.view(config.batch, config.sequence, config.heads, config.head_dim)
                return value.transpose(1, 2)

            query = rope(project(self.q_proj), cosine, sine)
            key = rope(project(self.k_proj), cosine, sine)
            value = project(self.v_proj)
            attended = functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True
            )
            attended = attended.transpose(1, 2).reshape(config.batch, config.sequence, config.hidden)
            hidden_states = residual + self.o_proj(attended)
            normalized = self.ffn_norm(hidden_states)
            feed_forward = functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
            return hidden_states + self.down_proj(feed_forward)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                config.vocab_size, config.hidden, device=device, dtype=target_dtype
            )
            self.layers = nn.ModuleList(DecoderBlock() for _ in range(config.layers))
            self.final_norm = RMSNorm()
            self.lm_head = nn.Linear(
                config.hidden, config.vocab_size, bias=False, device=device, dtype=target_dtype
            )
            if str(device) == "meta":
                cosine = torch.empty(
                    1, 1, config.sequence, config.head_dim // 2, device=device, dtype=target_dtype
                )
                sine = torch.empty_like(cosine)
            else:
                positions = torch.arange(config.sequence, device=device, dtype=torch.float32)
                frequencies = torch.arange(
                    0, config.head_dim, 2, device=device, dtype=torch.float32
                )
                frequencies = torch.pow(10000.0, -frequencies / config.head_dim)
                angles = positions[:, None] * frequencies[None, :]
                cosine = angles.cos().to(target_dtype)[None, None, :, :]
                sine = angles.sin().to(target_dtype)[None, None, :, :]
            self.register_buffer("rope_cosine", cosine)
            self.register_buffer("rope_sine", sine)

        def forward(self, token_ids: Any) -> Any:
            hidden_states = self.embedding(token_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states, self.rope_cosine, self.rope_sine)
            return self.lm_head(self.final_norm(hidden_states))

    return Model()


def acceptance_layers() -> tuple[tuple[int, float], ...]:
    return ((16, 0.815), (23, 1.145), (31, 1.522))
