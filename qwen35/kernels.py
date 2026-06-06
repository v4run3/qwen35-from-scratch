"""Optional fast kernels (FlashAttention, FLA) with torch fallbacks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_HAS_FLASH_ATTN = False
_HAS_FLA_GDN = False

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as fla_fused_recurrent_gated_delta_rule,
    )

    _HAS_FLA_GDN = True
except ImportError:
    fla_chunk_gated_delta_rule = None
    fla_fused_recurrent_gated_delta_rule = None


def flash_attention_available() -> bool:
    return _HAS_FLASH_ATTN


def fla_gdn_available() -> bool:
    return _HAS_FLA_GDN


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """q,k,v: [B, H, T, D]"""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        t = scores.shape[-1]
        k_len = scores.shape[-2]
        if t == k_len:
            mask = torch.triu(
                torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    return weights @ v


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    use_flash: bool = True,
) -> torch.Tensor:
    if use_flash and _HAS_FLASH_ATTN and q.is_cuda:
        # flash_attn expects [B, T, H, D]
        out = flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            causal=causal,
        )
        return out.transpose(1, 2)
    return eager_attention(q, k, v, causal=causal)
