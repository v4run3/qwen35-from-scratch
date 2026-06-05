from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qwen35.config import Qwen35Config
from qwen35.model import Qwen35ForCausalLM
from qwen35.training.bpe import BPETokenizer
from qwen35.training.data import TextLMDataset, load_text_corpus


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(
    *,
    data_path: str,
    output_dir: str,
    max_steps: int = 500,
    batch_size: int = 8,
    block_size: int = 128,
    learning_rate: float = 3e-4,
    warmup_steps: int = 50,
    eval_interval: int = 100,
    vocab_size: int = 4096,
    tokenizer_path: str | None = None,
    device: str | None = None,
    config: Qwen35Config | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    text = load_text_corpus(data_path)
    tokenizer_path_out = output / "tokenizer.json"

    if tokenizer_path:
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path} (vocab_size={tokenizer.vocab_size})")
    else:
        print(f"Training BPE tokenizer (target vocab_size={vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size)
        tokenizer.save(tokenizer_path_out)
        print(f"Saved tokenizer to {tokenizer_path_out} (vocab_size={tokenizer.vocab_size})")

    if config is None:
        config = Qwen35Config(
            hidden_size=256,
            intermediate_size=768,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=block_size,
            use_hybrid_attention=True,
            gdn_interval=4,
            use_moe=False,
        )
    config.vocab_size = tokenizer.vocab_size

    dataset = TextLMDataset(text, tokenizer, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Qwen35ForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Layer types: {config.layer_types}")
    print(f"Training on {len(dataset)} chunks, device={device}")

    step = 0
    model.train()
    while step < max_steps:
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            lr = get_lr(step, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            out = model(input_ids, labels=labels)
            assert out.loss is not None
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                print(f"step {step:5d} | loss {out.loss.item():.4f} | lr {lr:.2e}")

            if step > 0 and step % eval_interval == 0:
                model.save_pretrained(str(output / f"checkpoint_step_{step}.pt"))

            step += 1
            if step >= max_steps:
                break

    model.save_pretrained(str(output / "checkpoint_final.pt"))
    print(f"Saved final checkpoint to {output / 'checkpoint_final.pt'}")
