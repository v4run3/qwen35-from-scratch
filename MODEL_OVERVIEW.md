# Model Overview: Hybrid Qwen 3.5-Style Transformer

This document explains the implemented model in simple words, with the actual code structure and design decisions from the repository.

## What this project is

This repository builds a small transformer-based language model in PyTorch. It is a decoder-only model, which means it predicts the next word in a sequence using only the words that came before.

The main innovation is the hybrid use of two attention methods:

- **Gated DeltaNet (GDN)** linear attention for speed and long context
- **Full softmax attention** for high-quality reasoning every few layers

That combination is often called a **hybrid transformer**.

## Why hybrid attention?

A standard transformer uses full attention everywhere, which is powerful but slow when the input gets longer. A linear attention method like GDN is much faster for long sequences, but it can miss some detailed relationships.

So this model uses the fast GDN layers most of the time and then adds a full attention layer periodically. This gives a good balance:

- GDN = fast, efficient, long-range memory
- Full attention = rich reasoning, precise token interaction

## Main model structure

The model is built from repeating blocks. Each block contains several layers. In the current code, the pattern is:

- **Layer 1:** GDN linear attention
- **Layer 2:** GDN linear attention
- **Layer 3:** GDN linear attention
- **Layer 4:** Full softmax attention with Grouped Query Attention (GQA)

This repeats across the full depth of the model.

### What each layer does

- **Embedding layer** turns token IDs into vectors.
- **GDN layer** uses a special linear recurrence to compute attention without building a big attention matrix.
- **Full attention layer** uses a standard softmax attention formula, but with fewer key/value heads to save memory.
- **FFN / MLP layer** applies a feed-forward network with **SwiGLU** activation.
- **Final normalization** uses **RMSNorm** before the output head.

## Key pieces in easy words

### Gated DeltaNet (GDN)

GDN is a form of linear attention. Instead of comparing every token with every other token, it carries a running memory forward across the sequence.

Think of it like a train where each new car updates a shared state. That state keeps useful information from the whole sequence, so later tokens can read from it without scanning every earlier token.

This is fast and useful for long text.

### Full attention with GQA

Full attention is the classic transformer attention. It builds a score for every token pair and uses softmax to choose where to look.

This repo uses **Grouped Query Attention (GQA)**, which reduces the number of key/value heads. That means the model keeps full attention quality but uses less memory during generation.

### Rotary Positional Embeddings (RoPE)

RoPE gives the model a sense of token order. Instead of storing fixed position vectors, it rotates the query and key vectors depending on position.

This is a lightweight and stable way to tell the model which token comes first.

### RMSNorm

RMSNorm is a normalization layer that keeps the signal stable between residual blocks. It is simpler than LayerNorm and works well in transformer-style models.

### SwiGLU activation

SwiGLU is the activation function used inside the feed-forward network. It is a newer version of the common GELU/SWISH family and works better for many transformer models.

### Optional MoE (Mixture-of-Experts)

The code can optionally use a mixture-of-experts layer. That means the model can route each token through a small set of expert subnetworks instead of a single dense layer.

This is useful for increasing capacity without making every token pass through a huge dense layer.

## Training and tokenizer

### Tokenizer

The tokenizer is byte-level BPE, implemented from scratch in `qwen35/training/bpe.py`.

- It starts with 256 byte tokens plus a few special tokens
- It learns token merges based on the training text
- It can save and load `tokenizer.json`

### Training loop

Training happens in `qwen35/training/trainer.py` and `train.py`.

The process is:

1. Load raw text
2. Train or load the tokenizer
3. Convert text into chunks of token IDs
4. Build the model and optimizer
5. Train step-by-step with cross-entropy loss
6. Evaluate periodically using held-out text
7. Save checkpoints

The trainer also supports:

- mixed precision (FP16 / BF16)
- gradient checkpointing
- learning rate warmup and cosine decay

## Cache and generation support

The repository supports cached generation for fast autoregressive sampling:

- full attention layers use a KV cache for keys and values
- GDN layers store recurrent state and convolution state across tokens

This means the model does not recompute everything from scratch for every new token.

## How the code is organized

### Important files

- `qwen35/config.py` — model configuration and hybrid layer schedule
- `qwen35/model.py` — model wrapper and LM head
- `qwen35/layers.py` — layer factory for GDN and full attention blocks
- `qwen35/attention.py` — full softmax attention and RoPE
- `qwen35/gdn.py` — Gated DeltaNet, chunked/recurrent kernels, and short convolution
- `qwen35/mlp.py` — SwiGLU and optional MoE FFN
- `qwen35/norms.py` — RMSNorm and gated RMSNorm
- `qwen35/cache.py` — caching for KV and GDN state
- `qwen35/training/bpe.py` — byte-level tokenizer
- `qwen35/training/data.py` — dataset loading
- `qwen35/training/trainer.py` — training loop and evaluation
- `train.py` — training entrypoint
- `train_tokenizer.py` — tokenizer-only entrypoint

### Configuration highlights

`Qwen35Config` controls:

- number of layers
- hidden size and intermediate size
- number of heads and key/value head ratio
- whether hybrid mode is on
- how often full attention appears (`gdn_interval`)
- whether MoE is enabled
- mixed precision and gradient checkpointing

## Example hybrid block diagram

![Transformer overview](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Transformer.svg/1200px-Transformer.svg.png)

## Summary

This model is not a production-ready large language model yet. It is a research-style implementation of a hybrid transformer that mixes fast linear recurrence with periodic full softmax attention.

It is designed to be:

- easy to understand from code
- configurable through `Qwen35Config`
- trainable with a small text corpus
- efficient enough for longer contexts
- extensible for MoE and future kernel improvements

