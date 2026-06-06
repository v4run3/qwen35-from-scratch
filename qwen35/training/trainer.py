from __future__ import annotations

from contextlib import nullcontext
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


def resolve_amp_dtype(mode: str, device: str) -> torch.dtype | None:
    mode = mode.lower()
    device_type = torch.device(device).type
    if mode in ("none", "off", "false"):
        return None
    if device_type != "cuda":
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError("mixed_precision must be one of: none, auto, bf16, fp16")


def autocast_context(device: str, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype)


@torch.no_grad()
def evaluate(
    model: Qwen35ForCausalLM,
    loader: DataLoader,
    device: str,
    *,
    max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    for batch_idx, (input_ids, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with autocast_context(device, amp_dtype):
            out = model(input_ids, labels=labels)
        assert out.loss is not None
        losses.append(out.loss.item())

    if not losses:
        raise ValueError("Evaluation loader produced no batches")

    loss = sum(losses) / len(losses)
    perplexity = math.exp(min(loss, 50.0))
    model.train()
    return loss, perplexity


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
    eval_data_path: str | None = None,
    eval_batches: int | None = 20,
    vocab_size: int = 4096,
    tokenizer_path: str | None = None,
    mixed_precision: str = "none",
    gradient_checkpointing: bool = False,
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
    config.mixed_precision = mixed_precision
    config.gradient_checkpointing = gradient_checkpointing

    dataset = TextLMDataset(text, tokenizer, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_text = load_text_corpus(eval_data_path) if eval_data_path else text
    eval_dataset = TextLMDataset(eval_text, tokenizer, block_size=block_size)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = Qwen35ForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    amp_dtype = resolve_amp_dtype(mixed_precision, device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype == torch.float16 and torch.device(device).type == "cuda",
    )

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Layer types: {config.layer_types}")
    print(f"Mixed precision: {amp_dtype if amp_dtype is not None else 'off'}")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"Training on {len(dataset)} chunks, device={device}")
    print(f"Evaluating on {len(eval_dataset)} chunks")

    step = 0
    model.train()
    while step < max_steps:
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            lr = get_lr(step, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_dtype):
                out = model(input_ids, labels=labels)
                assert out.loss is not None

            if scaler.is_enabled():
                scaler.scale(out.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if step % 10 == 0:
                print(f"step {step:5d} | loss {out.loss.item():.4f} | lr {lr:.2e}")

            if step > 0 and step % eval_interval == 0:
                eval_loss, eval_ppl = evaluate(
                    model,
                    eval_loader,
                    device,
                    max_batches=eval_batches,
                    amp_dtype=amp_dtype,
                )
                print(
                    f"eval step {step:5d} | loss {eval_loss:.4f} | ppl {eval_ppl:.2f}"
                )
                model.save_pretrained(str(output / f"checkpoint_step_{step}.pt"))

            step += 1
            if step >= max_steps:
                break

    model.save_pretrained(str(output / "checkpoint_final.pt"))
    eval_loss, eval_ppl = evaluate(
        model,
        eval_loader,
        device,
        max_batches=eval_batches,
        amp_dtype=amp_dtype,
    )
    print(f"final eval | loss {eval_loss:.4f} | ppl {eval_ppl:.2f}")
    print(f"Saved final checkpoint to {output / 'checkpoint_final.pt'}")
