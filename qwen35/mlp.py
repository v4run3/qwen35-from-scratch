import torch
from torch import nn
import torch.nn.functional as F

from qwen35.config import Qwen35Config


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoESwiGLU(nn.Module):
    """Top-k routed experts plus always-on shared experts."""

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared = config.num_shared_experts
        hidden = config.hidden_size
        inter = config.moe_intermediate_size

        self.gate = nn.Linear(hidden, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden, inter) for _ in range(self.num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(hidden, inter) for _ in range(self.num_shared)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)

        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        top_weights = top_weights.to(flat.dtype)

        out = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (top_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            tokens = flat[mask]
            expert_out = expert(tokens)
            weight_sum = torch.zeros(tokens.shape[0], device=flat.device, dtype=flat.dtype)
            for k in range(self.top_k):
                slot_mask = top_indices[mask, k] == expert_idx
                weight_sum[slot_mask] += top_weights[mask, k][slot_mask]
            out[mask] += expert_out * weight_sum.unsqueeze(-1)

        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.reshape(batch, seq_len, hidden)


def build_mlp(config: Qwen35Config) -> nn.Module:
    if config.use_moe:
        return MoESwiGLU(config)
    return SwiGLU(config.hidden_size, config.intermediate_size)
