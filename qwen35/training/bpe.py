"""Byte-level BPE tokenizer (GPT-2 style), trained and used without external deps."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def _get_pairs(ids: tuple[int, ...]) -> set[tuple[int, int]]:
    return {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """
    Byte-level BPE: base vocab is 256 bytes, then learned merges, then special tokens.
    """

    VERSION = "bpe_v1"

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self._byte_offset = 4  # byte ids start at 4 (0-3 reserved for specials)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _init_byte_vocab(self) -> None:
        self.vocab = {
            self.pad_token_id: b"<|pad|>",
            self.unk_token_id: b"<|unk|>",
            self.bos_token_id: b"<|bos|>",
            self.eos_token_id: b"<|eos|>",
        }
        for b in range(256):
            self.vocab[self._byte_offset + b] = bytes([b])

    def train(self, text: str, vocab_size: int) -> None:
        min_vocab = self._byte_offset + 256  # 4 specials + 256 bytes
        if vocab_size < min_vocab:
            raise ValueError(f"vocab_size must be at least {min_vocab} (got {vocab_size})")

        self._init_byte_vocab()
        self.merges = {}
        self.merge_ranks = {}

        # Work in byte-token space (ids 4..259)
        tokens = [self._byte_offset + b for b in text.encode("utf-8")]
        next_id = self._byte_offset + 256

        while next_id < vocab_size:
            stats: Counter[tuple[int, int]] = Counter()
            for i in range(len(tokens) - 1):
                stats[(tokens[i], tokens[i + 1])] += 1
            if not stats:
                break

            pair = stats.most_common(1)[0][0]
            tokens = _merge_ids(tokens, pair, next_id)
            self.merges[pair] = next_id
            self.merge_ranks[pair] = len(self.merge_ranks)
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            next_id += 1

    def _encode_bytes(self, data: bytes) -> list[int]:
        ids = [self._byte_offset + b for b in data]
        while len(ids) >= 2:
            pairs = {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}
            ranked = [
                (p, self.merge_ranks[p]) for p in pairs if p in self.merge_ranks
            ]
            if not ranked:
                break
            pair = min(ranked, key=lambda x: x[1])[0]
            new_id = self.merges[pair]
            ids = _merge_ids(ids, pair, new_id)
        return ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._encode_bytes(text.encode("utf-8"))
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            piece = self.vocab.get(int(i))
            if piece is None:
                piece = self.vocab[self.unk_token_id]
            if piece in (b"<|pad|>", b"<|bos|>", b"<|eos|>"):
                continue
            if piece == b"<|unk|>":
                continue
            parts.append(piece)
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.VERSION,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "byte_offset": self._byte_offset,
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != cls.VERSION:
            raise ValueError(f"Unsupported tokenizer version: {data.get('version')}")

        tok = cls()
        tok.pad_token_id = data["pad_token_id"]
        tok.unk_token_id = data["unk_token_id"]
        tok.bos_token_id = data["bos_token_id"]
        tok.eos_token_id = data["eos_token_id"]
        tok._byte_offset = data["byte_offset"]
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        tok.merges = {}
        tok.merge_ranks = {}
        for rank, (a, b, nid) in enumerate(data["merges"]):
            pair = (int(a), int(b))
            tok.merges[pair] = int(nid)
            tok.merge_ranks[pair] = rank
        return tok

    @classmethod
    def train_from_file(
        cls, corpus_path: str | Path, vocab_size: int
    ) -> BPETokenizer:
        text = Path(corpus_path).read_text(encoding="utf-8")
        tok = cls()
        tok.train(text, vocab_size)
        return tok
