from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelCache:
    """KV cache for full attention + GDN recurrent/conv state for fast generation."""

    kv_keys: list[torch.Tensor | None]
    kv_values: list[torch.Tensor | None]
    gdn_states: list[torch.Tensor | None]
    gdn_conv_q: list[torch.Tensor | None]
    gdn_conv_k: list[torch.Tensor | None]
    gdn_conv_v: list[torch.Tensor | None]
    seen_tokens: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> ModelCache:
        n = [None] * num_layers
        return cls(
            kv_keys=list(n),
            kv_values=list(n),
            gdn_states=list(n),
            gdn_conv_q=list(n),
            gdn_conv_k=list(n),
            gdn_conv_v=list(n),
        )

    def seq_len(self) -> int:
        for k in self.kv_keys:
            if k is not None:
                return k.shape[2]
        for s in self.gdn_states:
            if s is not None:
                return s.shape[0] if s.dim() == 1 else s.shape[2] if s.dim() == 4 else 0
        for c in self.gdn_conv_q:
            if c is not None:
                return c.shape[-1] + 1
        return 0

    def has_previous_state(self) -> bool:
        return any(
            t is not None
            for group in (
                self.kv_keys,
                self.gdn_states,
                self.gdn_conv_q,
                self.gdn_conv_k,
                self.gdn_conv_v,
            )
            for t in group
        )

    def get_kv(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.kv_keys[layer_idx], self.kv_values[layer_idx]

    def update_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_k, past_v = self.get_kv(layer_idx)
        if past_k is None:
            self.kv_keys[layer_idx] = key
            self.kv_values[layer_idx] = value
            return key, value
        key = torch.cat([past_k, key], dim=2)
        value = torch.cat([past_v, value], dim=2)
        self.kv_keys[layer_idx] = key
        self.kv_values[layer_idx] = value
        return key, value

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.update_kv(layer_idx, key, value)

    def get_gdn_state(self, layer_idx: int) -> torch.Tensor | None:
        return self.gdn_states[layer_idx]

    def update_gdn_state(
        self,
        layer_idx: int,
        state: torch.Tensor | None,
    ) -> None:
        self.gdn_states[layer_idx] = state

    def get_gdn_conv_state(
        self,
        layer_idx: int,
        name: str,
    ) -> torch.Tensor | None:
        return self._gdn_conv_group(name)[layer_idx]

    def update_gdn_conv_state(
        self,
        layer_idx: int,
        name: str,
        state: torch.Tensor | None,
    ) -> None:
        self._gdn_conv_group(name)[layer_idx] = state

    def _gdn_conv_group(self, name: str) -> list[torch.Tensor | None]:
        if name == "q":
            return self.gdn_conv_q
        if name == "k":
            return self.gdn_conv_k
        if name == "v":
            return self.gdn_conv_v
        raise ValueError(f"Unknown GDN conv state: {name}")


# Backward-compatible alias
KVCache = ModelCache
