from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    """Per-layer key/value cache for full (softmax) attention during generation."""

    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]

    @classmethod
    def empty(cls, num_layers: int) -> KVCache:
        return cls(
            keys=[None] * num_layers,
            values=[None] * num_layers,
        )

    def seq_len(self) -> int:
        for k in self.keys:
            if k is not None:
                return k.shape[2]
        return 0

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.keys[layer_idx], self.values[layer_idx]

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_k, past_v = self.get(layer_idx)
        if past_k is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
            return key, value
        key = torch.cat([past_k, key], dim=2)
        value = torch.cat([past_v, value], dim=2)
        self.keys[layer_idx] = key
        self.values[layer_idx] = value
        return key, value
