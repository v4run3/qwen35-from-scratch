# qwen35-from-scratch

A from-scratch implementation of a **Qwen 3.5–style** decoder in PyTorch: hybrid **Gated DeltaNet** + full attention, optional **MoE** FFN, **byte-level BPE**, training, and KV-cache generation.

## Architecture (implemented)

| Component | Status |
|-----------|--------|
| RMSNorm, RoPE, GQA full attention | Done |
| **Gated DeltaNet** (chunk + recurrent fallback) | Done |
| **3:1 hybrid** (GDN × 3, full attention × 1 per `gdn_interval`) | Done |
| SwiGLU dense FFN | Done |
| **MoE** FFN (top-k + shared experts) | Done (`--moe`) |
| Tied embeddings / LM head | Done |
| KV-cache generation | Done |
| **Byte-level BPE tokenizer** | Done |
| Training loop | Done |

Not yet: multimodal vision, official weight loading, FlashAttention / FLA kernels, distributed training.

## Project layout

```
qwen35/
  config.py, cache.py, rope.py, norms.py, utils.py
  attention.py, gdn.py, mlp.py, layers.py, model.py, generation.py
  training/
    bpe.py          # BPE train / encode / decode
    data.py, trainer.py
train.py            # train model + BPE on corpus
train_tokenizer.py  # train BPE only
```

## Quick start

```bash
pip install -r requirements.txt
python -m qwen35
```

Train BPE only:

```bash
python train_tokenizer.py --data data/sample.txt --output runs/demo/tokenizer.json --vocab-size 512
```

Train model (fits BPE on corpus, saves `tokenizer.json` in output dir):

```bash
python train.py --data data/sample.txt --output-dir runs/demo --max-steps 200 --vocab-size 512 --no-hybrid
```

Reuse a trained tokenizer:

```bash
python train.py --data data/sample.txt --output-dir runs/demo --tokenizer runs/demo/tokenizer.json
```

Options:

- `--vocab-size` — target BPE vocabulary size (default 4096)
- `--no-hybrid` — all layers use full attention (faster on CPU)
- `--moe` — mixture-of-experts FFN

## BPE format

`tokenizer.json` stores merge rules and a byte vocabulary (special tokens + 256 bytes + merged tokens). Encoding is greedy merge by rank; decoding concatenates token bytes as UTF-8.

## Roadmap

1. **Gradient checkpointing** — larger models on one GPU
2. **GDN inference cache** — fast single-token decode through linear layers
3. **Eval harness** — perplexity on held-out text

## Reference

GDN recurrence follows torch fallbacks in [Hugging Face Qwen3-Next](https://github.com/huggingface/transformers) / OLMo Hybrid (Apache 2.0).
