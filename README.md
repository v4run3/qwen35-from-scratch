# qwen35-from-scratch

A from-scratch implementation of a Qwen 3.5-style decoder in PyTorch: hybrid
Gated DeltaNet plus full attention, optional MoE FFN, byte-level BPE, training,
evaluation, and cached generation.

## Architecture

| Component | Status |
|-----------|--------|
| RMSNorm, RoPE, GQA full attention | Done |
| Gated DeltaNet chunk + recurrent fallback | Done |
| 3:1 hybrid layers via `gdn_interval` | Done |
| SwiGLU dense FFN | Done |
| MoE FFN | Done (`--moe`) |
| Tied embeddings / LM head | Done |
| Full-attention KV cache | Done |
| GDN inference cache | Done |
| Byte-level BPE tokenizer | Done |
| Training loop | Done |
| Eval loss / perplexity | Done |
| Mixed precision training | Done (`--mixed-precision`) |
| Gradient checkpointing | Done (`--gradient-checkpointing`) |

Not yet: multimodal vision, official weight loading, distributed training, and
production FlashAttention / FLA integration.

## Project Layout

```text
qwen35/
  config.py, cache.py, rope.py, norms.py, utils.py
  attention.py, gdn.py, mlp.py, layers.py, model.py, generation.py
  training/
    bpe.py          # BPE train / encode / decode
    data.py, trainer.py
train.py            # train model + BPE on corpus
train_tokenizer.py  # train BPE only
```

## Quick Start

```bash
pip install -r requirements.txt
python -m qwen35
```

Train BPE only:

```bash
python train_tokenizer.py --data data/sample.txt --output runs/demo/tokenizer.json --vocab-size 512
```

Train model:

```bash
python train.py --data data/sample.txt --output-dir runs/demo --max-steps 200 --vocab-size 512 --no-hybrid
```

Reuse a trained tokenizer:

```bash
python train.py --data data/sample.txt --output-dir runs/demo --tokenizer runs/demo/tokenizer.json
```

Useful options:

- `--vocab-size`: target BPE vocabulary size, default 4096
- `--no-hybrid`: all layers use full attention, faster on CPU
- `--moe`: enable mixture-of-experts FFN
- `--eval-data`: optional held-out text file for perplexity
- `--eval-batches`: cap validation batches for faster checks
- `--mixed-precision {none,auto,bf16,fp16}`: CUDA autocast mode
- `--gradient-checkpointing`: reduce activation memory while training

## BPE Format

`tokenizer.json` stores merge rules and a byte vocabulary: special tokens, 256
raw byte tokens, and merged tokens. Encoding is greedy merge by rank; decoding
concatenates token bytes as UTF-8.

## Roadmap

1. Official checkpoint/config conversion
2. Longer-context training polish
3. Optional FlashAttention / FLA integration

## Reference

GDN recurrence follows torch fallbacks in Hugging Face Qwen3-Next and OLMo
Hybrid style implementations.
