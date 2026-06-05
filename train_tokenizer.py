"""Train or inspect a BPE tokenizer without running full model training."""

from __future__ import annotations

import argparse

from qwen35.training.bpe import BPETokenizer
from qwen35.training.data import load_text_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on a text file")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--encode", type=str, default=None, help="Optional string to encode/decode")
    args = parser.parse_args()

    text = load_text_corpus(args.data)
    tokenizer = BPETokenizer()
    tokenizer.train(text, args.vocab_size)
    tokenizer.save(args.output)
    print(f"Saved BPE tokenizer to {args.output} (vocab_size={tokenizer.vocab_size})")

    if args.encode:
        ids = tokenizer.encode(args.encode)
        print("ids:", ids[:32], ("..." if len(ids) > 32 else ""))
        print("decoded:", repr(tokenizer.decode(ids)))


if __name__ == "__main__":
    main()
