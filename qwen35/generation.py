from __future__ import annotations

import torch
import torch.nn.functional as F

from qwen35.cache import KVCache
from qwen35.layers import DecoderLayer
from qwen35.model import Qwen35ForCausalLM


@torch.no_grad()
def generate(
    model: Qwen35ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    model.eval()
    config = model.config

    has_full_attn = any(
        isinstance(layer, DecoderLayer) for layer in model.model.layers
    )
    kv_cache = (
        KVCache.empty(config.num_hidden_layers)
        if use_kv_cache and has_full_attn
        else None
    )

    out = model(input_ids, kv_cache=kv_cache, position_offset=0)

    for step in range(max_new_tokens):
        next_token_logits = out.logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled = next_token_logits / temperature
            if top_k is not None:
                top_values, _ = torch.topk(scaled, top_k)
                scaled = scaled.masked_fill(
                    scaled < top_values[:, [-1]],
                    torch.finfo(scaled.dtype).min,
                )
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

        if step == max_new_tokens - 1:
            break

        if kv_cache is not None:
            out = model(
                input_ids[:, -1:],
                kv_cache=kv_cache,
                position_offset=input_ids.shape[1] - 1,
            )
        else:
            out = model(input_ids)

    return input_ids
