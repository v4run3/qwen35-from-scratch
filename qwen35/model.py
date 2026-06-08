from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
            if self.config.gradient_checkpointing and self.training and kv_cache is None:
                x = checkpoint(
                    lambda hidden, block=layer: block(
                        hidden,
                        kv_cache=None,
                        position_offset=0,
                    ),
                    x,
                    use_reentrant=False,
                )
            elif isinstance(layer, DecoderLayer):
                x = layer(
                    x,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                )
            else:
                x = layer(
                    x,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                )
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

    def save_pretrained_hf(self, directory: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            import dataclasses
            json.dump(dataclasses.asdict(self.config), f, indent=2)
        weights_path = directory / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)
        print(f"Saved HF-format checkpoint to {directory}")

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> Qwen35ForCausalLM:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)

    @classmethod
    def from_pretrained_hf(cls, directory: str, device: str = "cpu") -> Qwen35ForCausalLM:
        directory = Path(directory)
        config_path = directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {directory}")
        with open(config_path, "r", encoding="utf-8") as f:
            import dataclasses
            config_dict = json.load(f)
        config = Qwen35Config(**config_dict)

        weights_path = directory / "pytorch_model.bin"
        if not weights_path.exists():
            weights_path = directory / "model.safetensors"
            if not weights_path.exists():
                raise FileNotFoundError(f"Neither pytorch_model.bin nor model.safetensors found in {directory}")

        if str(weights_path).endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_path), device=device)
            except ImportError:
                raise ImportError("safetensors package required to load .safetensors files. Install with: pip install safetensors")
        else:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)

        model = cls(config)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device)
