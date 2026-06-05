from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import Dataset

from qwen35.training.bpe import BPETokenizer


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    @property
    def vocab_size(self) -> int: ...


class TextLMDataset(Dataset):
    """Sliding-window chunks from tokenized text."""

    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        block_size: int = 128,
        stride: int | None = None,
    ):
        self.block_size = block_size
        self.stride = stride or block_size
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"Corpus too short ({len(self.data)} tokens) for block_size={block_size}. "
                "Use a larger file or smaller --block-size."
            )
        self.num_chunks = (len(self.data) - block_size - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_text_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
