import torch
from torch import nn
import torch.nn.functional as F

from qwen35.cache import KVCache
from qwen35.config import Qwen35Config
from qwen35.rope import RotaryEmbedding, apply_rotary_pos_emb
from qwen35.utils import repeat_kv


class SelfAttention(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
        position_offset: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len, x.device, offset=position_offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
            key_len = k.shape[2]
        else:
            key_len = seq_len

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim**0.5)

        if kv_cache is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask, torch.finfo(x.dtype).min
            )
        else:
            # Query positions attend to all cached keys (causal by construction)
            pass

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        return self.o_proj(context)
