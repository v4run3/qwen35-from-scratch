import torch
from torch import nn

from qwen35.attention import SelfAttention
from qwen35.cache import KVCache
from qwen35.config import Qwen35Config
from qwen35.gdn import GatedDeltaNet
from qwen35.mlp import build_mlp
from qwen35.norms import RMSNorm


class DecoderLayer(nn.Module):
    """Full softmax attention + dense or MoE FFN."""

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = build_mlp(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            position_offset=position_offset,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GDNLayer(nn.Module):
    """Gated DeltaNet linear attention + dense or MoE FFN."""

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.linear_attn = GatedDeltaNet(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = build_mlp(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = x + self.linear_attn(self.input_layernorm(x), kv_cache=kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def build_decoder_layer(
    config: Qwen35Config, layer_idx: int, layer_type: str
) -> nn.Module:
    if layer_type == "gdn":
        return GDNLayer(config, layer_idx)
    if layer_type == "full_attention":
        return DecoderLayer(config, layer_idx)
    raise ValueError(f"Unknown layer type: {layer_type}")
