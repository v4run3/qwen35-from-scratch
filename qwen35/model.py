from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from qwen35.cache import KVCache
from qwen35.config import Qwen35Config
from qwen35.layers import DecoderLayer, build_decoder_layer
from qwen35.norms import RMSNorm


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TextModel(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        assert config.layer_types is not None
        self.layers = nn.ModuleList(
            [
                build_decoder_layer(config, i, config.layer_types[i])
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_offset: int | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if position_offset is None:
            position_offset = kv_cache.seq_len() if kv_cache is not None else 0

        for layer in self.layers:
            if isinstance(layer, DecoderLayer):
                x = layer(
                    x,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                )
            else:
                x = layer(x)
        return self.norm(x)


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        position_offset: int | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.model(
            input_ids,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str) -> None:
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> Qwen35ForCausalLM:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)
