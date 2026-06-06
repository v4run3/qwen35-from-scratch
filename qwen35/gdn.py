"""Gated DeltaNet linear attention (reference: Qwen3-Next / OLMo Hybrid torch fallbacks)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from qwen35.cache import KVCache
from qwen35.config import Qwen35Config
from qwen35.norms import RMSNormGated
from qwen35.utils import l2norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = last_recurrent_state * g[:, :, i, -1, None, None].exp() + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class ShortConvolution(nn.Module):
    """Depthwise causal conv on sequence (local n-gram context before GDN)."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        x_t = x.transpose(1, 2)
        if conv_state is None:
            y = self.conv(x_t)
            y = y[..., : x_t.shape[-1]]
            full = x_t
        else:
            full = torch.cat((conv_state.to(x_t), x_t), dim=-1)
            if full.shape[-1] < self.kernel_size:
                full = F.pad(full, (self.kernel_size - full.shape[-1], 0))
            y = F.conv1d(full, self.conv.weight, groups=self.conv.groups)

        y = F.silu(y.transpose(1, 2))
        if not return_state:
            return y

        if self.kernel_size == 1:
            new_state = x_t[..., :0]
        else:
            state_source = full
            if state_source.shape[-1] < self.kernel_size - 1:
                state_source = F.pad(
                    state_source,
                    (self.kernel_size - 1 - state_source.shape[-1], 0),
                )
            new_state = state_source[..., -(self.kernel_size - 1) :].detach()
        return y, new_state


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.layer_idx = layer_idx
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        self.chunk_size = config.gdn_chunk_size

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        ks = config.linear_conv_kernel_dim
        self.q_conv = ShortConvolution(self.key_dim, ks)
        self.k_conv = ShortConvolution(self.key_dim, ks)
        self.v_conv = ShortConvolution(self.value_dim, ks)

        a = torch.empty(self.num_v_heads).uniform_(
            config.linear_a_log_min, config.linear_a_log_max
        )
        self.A_log = nn.Parameter(torch.log(a))

        dt = torch.exp(
            torch.rand(self.num_v_heads)
            * (math.log(config.linear_dt_max) - math.log(config.linear_dt_min))
            + math.log(config.linear_dt_min)
        )
        dt = torch.clamp(dt, min=config.linear_dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.o_norm = RMSNormGated(self.head_v_dim)

    def _conv_forward(
        self,
        conv: ShortConvolution,
        x: torch.Tensor,
        kv_cache: KVCache | None,
        name: str,
    ) -> torch.Tensor:
        if kv_cache is None:
            out = conv(x)
            assert isinstance(out, torch.Tensor)
            return out

        conv_state = kv_cache.get_gdn_conv_state(self.layer_idx, name)
        out, new_state = conv(x, conv_state=conv_state, return_state=True)
        kv_cache.update_gdn_conv_state(self.layer_idx, name, new_state)
        return out

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self._conv_forward(self.q_conv, self.q_proj(x), kv_cache, "q")
        k = self._conv_forward(self.k_conv, self.k_proj(x), kv_cache, "k")
        v = self._conv_forward(self.v_conv, self.v_proj(x), kv_cache, "v")

        q = q.view(batch_size, seq_len, -1, self.head_k_dim)
        k = k.view(batch_size, seq_len, -1, self.head_k_dim)
        v = v.view(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads > self.num_k_heads:
            ratio = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(ratio, dim=2)
            k = k.repeat_interleave(ratio, dim=2)

        beta = self.b_proj(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias
        )

        initial_state = kv_cache.get_gdn_state(self.layer_idx) if kv_cache is not None else None
        output_final_state = kv_cache is not None

        if seq_len == 1:
            output, final_state = torch_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            output, final_state = torch_chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g,
                beta=beta,
                chunk_size=self.chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=True,
            )
        if kv_cache is not None:
            kv_cache.update_gdn_state(self.layer_idx, final_state)

        gate = self.g_proj(x)
        output = output.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        output = self.o_norm(output, gate)
        output = output.reshape(batch_size, seq_len, -1)
        return self.o_proj(output)
