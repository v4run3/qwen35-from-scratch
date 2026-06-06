from __future__ import annotations

from dataclasses import dataclass


def build_layer_types(num_hidden_layers: int, gdn_interval: int = 4) -> list[str]:
    """3 GDN layers then 1 full attention (Qwen 3.5-style hybrid)."""
    layer_types = ["gdn"] * num_hidden_layers
    for i in range(num_hidden_layers):
        if (i + 1) % gdn_interval == 0:
            layer_types[i] = "full_attention"
    if "full_attention" not in layer_types:
        layer_types[-1] = "full_attention"
    return layer_types


@dataclass
class Qwen35Config:
    vocab_size: int = 32000
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 1024
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: str = "none"

    # Hybrid attention (GDN + full softmax every gdn_interval layers)
    use_hybrid_attention: bool = True
    gdn_interval: int = 4
    layer_types: list[str] | None = None

    # Gated DeltaNet (linear attention) head dims
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int = 4
    linear_a_log_min: float = 0.0
    linear_a_log_max: float = 16.0
    linear_dt_min: float = 0.001
    linear_dt_max: float = 0.1
    linear_dt_init_floor: float = 1e-4
    linear_allow_neg_eigval: bool = True
    gdn_chunk_size: int = 64

    # Mixture-of-experts FFN
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int | None = None

    def __post_init__(self) -> None:
        if self.linear_num_key_heads is None:
            self.linear_num_key_heads = self.num_attention_heads
        if self.linear_num_value_heads is None:
            self.linear_num_value_heads = self.num_attention_heads
        if self.linear_key_head_dim is None:
            self.linear_key_head_dim = int(
                0.75 * self.hidden_size / self.linear_num_key_heads
            )
        if self.linear_value_head_dim is None:
            self.linear_value_head_dim = 2 * self.linear_key_head_dim
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        if self.use_hybrid_attention and self.layer_types is None:
            self.layer_types = build_layer_types(
                self.num_hidden_layers, self.gdn_interval
            )
        elif not self.use_hybrid_attention:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
