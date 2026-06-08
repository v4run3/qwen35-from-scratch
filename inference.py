"""Standalone inference / generation script for a saved Qwen35 checkpoint."""

from __future__ import annotations

import argparse
import os

import torch

from qwen35.cache import KVCache
from qwen35.generation import generate
from qwen35.model import Qwen35ForCausalLM
from qwen35.training.bpe import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a saved Qwen35 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer.json (defaults to same dir as checkpoint)")
    parser.add_argument("--prompt", type=str, default="", help="Input prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV/state cache for debugging")
    parser.add_argument("--flash", action="store_true", help="Force FlashAttention/FLA kernels when available")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint
    ckpt_dir = os.path.dirname(ckpt_path) or "."
    tok_path = args.tokenizer or os.path.join(ckpt_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Pass --tokenizer or ensure tokenizer.json is in the checkpoint directory.")

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Qwen35ForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if args.flash:
        from qwen35.kernels import flash_attention_available, fla_gdn_available
        print(f"FlashAttention available: {flash_attention_available()}")
        print(f"FLA GDN available: {fla_gdn_available()}")

    tokenizer = BPETokenizer.load(tok_path)
    prompt_text = args.prompt or input("Enter prompt: ")

    input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)

    print(f"Prompt: {repr(prompt_text)}")
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Config layer_types: {config.layer_types}")

    generated = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_kv_cache=not args.no_kv_cache,
    )
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"\nGenerated:\n{decoded}")


if __name__ == "__main__":
    main()
