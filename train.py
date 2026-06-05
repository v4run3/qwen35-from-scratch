"""Train a small Qwen 3.5-style model on a text file."""

from __future__ import annotations

import argparse

from qwen35.config import Qwen35Config
from qwen35.training.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen35 from scratch")
    parser.add_argument("--data", type=str, required=True, help="Path to UTF-8 text corpus")
    parser.add_argument("--output-dir", type=str, default="runs/default")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--vocab-size", type=int, default=4096, help="BPE vocab size to train")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to existing tokenizer.json (skip BPE training)",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--no-hybrid", action="store_true", help="Use only full attention")
    parser.add_argument("--moe", action="store_true", help="Enable MoE FFN")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Qwen35Config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.block_size,
        use_hybrid_attention=not args.no_hybrid,
        use_moe=args.moe,
        num_experts=args.num_experts,
    )
    train(
        data_path=args.data,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        vocab_size=args.vocab_size,
        tokenizer_path=args.tokenizer,
        device=args.device,
        config=config,
    )


if __name__ == "__main__":
    main()
