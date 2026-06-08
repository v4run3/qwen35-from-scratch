# qwen35-from-scratch

A from-scratch implementation of a Qwen 3.5-style decoder in PyTorch. The model mixes
fast **Gated DeltaNet (GDN)** linear attention with periodic **full softmax attention**
to balance long-context efficiency and reasoning quality. It also includes optional
**Mixture-of-Experts (MoE)** feed-forward layers, byte-level BPE tokenization,
mixed-precision training, gradient checkpointing, FlashAttention/FLA acceleration,
distributed training, and HuggingFace-format weight loading.

---

## How the model works (high level)

1. **Text → tokens**: raw text is encoded into integer token IDs by a byte-level BPE
   tokenizer.
2. **Tokens → vectors**: each ID is looked up in an embedding table to produce a
   hidden-state vector.
3. **Layer stack**: the vector passes through a sequence of transformer blocks. Most
   blocks are **GDN layers** (linear recurrence), and every *gdn_interval* layers a
   **full-attention layer** is inserted instead.
4. **Output**: after the final normalization, a linear head maps hidden states to
   vocabulary logits, which are used to predict the next token.

This hybrid design means the model spends most of its compute on fast, O(1)-per-token
GDN layers, while still getting the strong local reasoning of full softmax attention
periodically.

---

## Component deep-dive

### 1. Tokenizer — byte-level BPE
**File**: `qwen35/training/bpe.py`

The tokenizer learns a vocabulary of subword units from raw text. It starts with 256
raw byte tokens plus a small number of special tokens, then greedily merges the most
frequent adjacent pairs. Encoding converts text → token IDs; decoding converts token
IDs → bytes → text. The trained tokenizer can be saved to and loaded from
`tokenizer.json`.

### 2. Embedding layer
**File**: `qwen35/model.py:26`

`nn.Embedding(vocab_size, hidden_size)` maps each token ID to a dense vector. This is
the first layer the input passes through. If `tie_word_embeddings=True` (the default),
the final language-model head shares this same weight matrix, which saves memory and
keeps the embedding and output spaces aligned.

### 3. Normalization — RMSNorm
**File**: `qwen35/norms.py:6`

**RMSNorm** (`RMSNorm`) normalizes each token's hidden vector by its root-mean-square
value, then applies a learnable scale (`weight`). It is simpler than LayerNorm and is
used as the pre-normalization before every attention and feed-forward sub-layer.

**RMSNormGated** (`RMSNormGated`, `qwen35/norms.py:17`) is a variant used at the
output of GDN layers. It first applies RMSNorm, then multiplies by a `SiLU`-gated
signal so the layer can adaptively modulate its output.

### 4. Positional encoding — RoPE
**File**: `qwen35/rope.py:5`

**Rotary Position Embeddings (RoPE)** give the model a sense of token order without
adding extra parameters to the embedding table. Each query and key vector is rotated
by position-dependent angles before the attention score is computed. The rotation is
applied in pairs of dimensions, which preserves the vector's length and makes the
relative position between two tokens easy to compute. `RotaryEmbedding` precomputes
the frequency table; `apply_rotary_pos_emb` does the actual rotation.

### 5. Full attention — Grouped Query Attention (GQA)
**File**: `qwen35/attention.py:11`

When a layer is a "full attention" layer, it uses standard softmax attention with
**Grouped Query Attention**:

- **Q** (queries) has `num_attention_heads` heads.
- **K** and **V** (keys and values) have fewer heads: `num_key_value_heads`.
- The KV heads are **repeated** (via `repeat_kv` in `qwen35/utils.py:4`) so every
  query head has a key/value to attend to.

This reduces the size of the KV cache during generation while keeping attention
quality high. The actual attention computation is delegated to
`qwen35/kernels.py:60` (`attention_forward`), which will automatically use
**FlashAttention** if the library is installed and the input is on CUDA; otherwise it
falls back to a pure-PyTorch eager implementation.

### 6. Linear attention — Gated DeltaNet (GDN)
**File**: `qwen35/gdn.py:204`

GDN is a **linear-attention** mechanism. Instead of computing an N×N attention matrix
(which costs O(N²) memory and time), GDN maintains a compact **recurrent state** that
is updated token by token. This gives O(1) per-token cost during generation and O(N)
cost during training when using the chunked kernel.

Key sub-components inside `GatedDeltaNet`:

- **Projections** (`qwen35/gdn.py:218-223`): `q_proj`, `k_proj`, `v_proj` map the
  input to query, key, and value spaces. `a_proj` and `b_proj` produce the **decay
  (`g`)** and **gate (`beta`)** signals that control the recurrence.
- **Short convolution** (`qwen35/gdn.py:154`): before entering the recurrence, Q, K,
  and V each pass through a tiny depthwise causal 1-D convolution (`ShortConvolution`).
  This gives the linear layer a small local look-back window (default kernel size 4)
  so it can capture immediate n-gram patterns before the global accumulation begins.
- **Delta rule recurrence**: the core recurrence updates a state matrix
  `last_recurrent_state` using the current key, value, decay, and gate. The output
  for a token is computed by reading from this state with the current query.
- **Chunked vs. recurrent mode** (`qwen35/gdn.py:294-354`):
  - If `seq_len == 1` (generation), the **fused recurrent** kernel is used. If the
    **FLA** library is installed and CUDA is available, `fla_fused_recurrent_gated_delta_rule`
    is called; otherwise the pure-torch `torch_recurrent_gated_delta_rule` is used.
  - If `seq_len > 1` (training / prefill), the **chunked** kernel is used. It splits
    the sequence into chunks of size `gdn_chunk_size` (default 64), processes each
    chunk with a blocked algorithm, and carries the recurrent state across chunk
    boundaries. Again, FLA's `fla_chunk_gated_delta_rule` is used when available.

### 7. Feed-forward networks — SwiGLU and MoE
**File**: `qwen35/mlp.py:8`

Inside every transformer block, after the attention sub-layer, the hidden states pass
through a feed-forward network (FFN).

- **SwiGLU** (`SwiGLU`, `qwen35/mlp.py:8`): the standard FFN used in most layers. It
  projects the hidden state up to a larger `intermediate_size`, applies
  `SiLU(gate_proj(x)) * up_proj(x)`, then projects back down with `down_proj`. This
  activation pattern is more expressive than plain GELU and is used in models like
  LLaMA and Qwen.
- **MoESwiGLU** (`MoESwiGLU`, `qwen35/mlp.py:19`): an optional mixture-of-experts
  variant. A router (`gate`) selects the top-k experts for each token. Each expert is
  itself a small `SwiGLU` network. Additionally, **shared experts** (always active)
  ensure every token has a base level of processing capacity. This increases model
  capacity without proportionally increasing per-token compute.

The builder function `build_mlp` (`qwen35/mlp.py:67`) chooses between `SwiGLU` and
`MoESwiGLU` based on `config.use_moe`.

### 8. Layer stacking — hybrid block pattern
**Files**: `qwen35/layers.py`, `qwen35/config.py:6`

The model does not use a single attention type everywhere. Instead, it arranges layers
in a repeating **hybrid block**:

- By default, every `gdn_interval`-th layer (default every 4th) is a **full attention**
  layer (`DecoderLayer`).
- All other layers are **GDN layers** (`GDNLayer`).

The schedule is built by `build_layer_types` (`qwen35/config.py:6`). For example,
with `num_hidden_layers=8` and `gdn_interval=4`, the pattern is:
`[gdn, gdn, gdn, full_attention, gdn, gdn, gdn, full_attention]`.

Both layer types share the same structure:
`x → RMSNorm → attention/GDN → residual → RMSNorm → FFN → residual`.

### 9. Caching — KV cache and GDN state
**File**: `qwen35/cache.py:9`

During autoregressive generation, the model avoids recomputing everything from scratch
for each new token by caching intermediate results in a `ModelCache` object:

- **KV cache** (`kv_keys`, `kv_values`): full-attention layers cache the key and value
  projections for every past token. On the next step, only the new token's KV pair is
  computed and appended.
- **GDN recurrent state** (`gdn_states`): each GDN layer stores its compact recurrent
  matrix so the next token can be processed in O(1).
- **GDN convolution state** (`gdn_conv_q`, `gdn_conv_k`, `gdn_conv_v`): each
  `ShortConvolution` keeps the last `kernel_size-1` inputs so the causal convolution
  can continue streaming without re-convolving the whole prefix.

The cache is initialized empty with `KVCache.empty(num_layers)` and passed through
every forward call during generation.

### 10. Generation
**File**: `qwen35/generation.py:10`

`generate()` runs autoregressive text generation. Given a prompt and a model:
1. Run the prompt through the model once with the cache enabled to "prime" it.
2. Repeatedly sample the next token from the output logits (with optional temperature
   and top-k filtering).
3. Append the new token to the input and run just that single new token through the
   model, updating the cache at each step.

The function returns the full sequence of token IDs, which can be decoded back to
text with the tokenizer.

### 11. Training loop
**Files**: `qwen35/training/trainer.py`, `train.py`

The training pipeline:
1. Load a raw text corpus.
2. Train or load a BPE tokenizer.
3. Chunk the tokenized text into fixed-length sequences (with a sliding window).
4. Build the model and an AdamW optimizer.
5. Loop over batches:
   - Compute a cosine-decay learning rate with warmup.
   - Run a forward pass to get cross-entropy loss on the next-token prediction task.
   - Backpropagate, clip gradients, and step the optimizer.
6. Periodically evaluate on a held-out set to measure **perplexity** (`math.exp(loss)`).
7. Save checkpoints.

The trainer supports mixed-precision autocast (BF16/FP16), `torch.cuda.amp.GradScaler`
for FP16 stability, gradient checkpointing (`torch.utils.checkpoint.checkpoint`) to
trade compute for memory, and **DistributedDataParallel** for multi-GPU training.

### 12. Inference script
**File**: `inference.py`

A standalone CLI for running a saved checkpoint. It loads the model and tokenizer,
prompts for input text, and prints the generated continuation. Flags control
`--max-new-tokens`, `--temperature`, `--top-k`, `--device`, and `--flash`.

### 13. Weight loading and saving
**Files**: `qwen35/model.py:109-160`

- `save_pretrained(path)` saves a `.pt` file containing both the config and the
  model's `state_dict`.
- `save_pretrained_hf(directory)` writes a HuggingFace-compatible directory with
  `config.json` and `pytorch_model.bin`.
- `from_pretrained(path)` loads the `.pt` format.
- `from_pretrained_hf(directory)` loads a HuggingFace-format directory, supporting
  both `.bin` and `.safetensors` weights (the latter requires the `safetensors`
  package).

### 14. FlashAttention and FLA acceleration
**File**: `qwen35/kernels.py`

`attention_forward()` is a thin wrapper used by the full-attention layer. If
`flash-attn` is installed and the tensors are on CUDA, it calls `flash_attn_func`
for a fused, memory-efficient attention kernel. Otherwise it falls back to eager
PyTorch attention.

Similarly, the GDN layer in `qwen35/gdn.py` will call **FLA** (Flash Linear
Attention) kernels (`fla_chunk_gated_delta_rule` and
`fla_fused_recurrent_gated_delta_rule`) when available on CUDA. If the libraries are
not installed or the model is on CPU, it silently uses the pure-torch implementations
defined in the same file.

---

## Configuration
**File**: `qwen35/config.py:18`

`Qwen35Config` is a dataclass that controls every architectural and training knob:

| Parameter | Meaning |
|-----------|---------|
| `vocab_size` | Number of token types in the BPE vocabulary |
| `hidden_size` | Dimension of the hidden states throughout the model |
| `intermediate_size` | FFN inner dimension (dense SwiGLU) |
| `num_hidden_layers` | Total number of transformer blocks |
| `num_attention_heads` | Query heads in full-attention layers |
| `num_key_value_heads` | KV heads in full-attention layers (GQA) |
| `max_position_embeddings` | Maximum sequence length the model was trained for |
| `rope_theta` | Base frequency for RoPE |
| `rms_norm_eps` | Epsilon in RMSNorm denominator for numerical stability |
| `tie_word_embeddings` | Share embedding weights with the LM head |
| `gradient_checkpointing` | Recompute activations to save memory |
| `mixed_precision` | `none`, `auto`, `bf16`, or `fp16` |
| `use_hybrid_attention` | Toggle GDN + full-attention hybrid |
| `gdn_interval` | Insert a full-attention layer every N layers |
| `linear_num_key_heads` / `linear_num_value_heads` | Heads inside GDN |
| `linear_key_head_dim` / `linear_value_head_dim` | Per-head dims inside GDN |
| `linear_conv_kernel_dim` | Short-conv kernel size before GDN recurrence |
| `linear_a_log_min/max` | Range for the learnable decay log-parameter |
| `linear_dt_min/max` | Range for the learnable delta-bias initialization |
| `linear_allow_neg_eigval` | Allow negative eigenvalues in GDN (doubles beta range) |
| `gdn_chunk_size` | Chunk size for the chunked GDN kernel |
| `use_moe` | Enable MoE FFN instead of dense SwiGLU |
| `num_experts` | Total number of MoE experts |
| `num_experts_per_tok` | Top-k experts selected per token |
| `num_shared_experts` | Experts always applied to every token |
| `moe_intermediate_size` | Hidden size inside each MoE expert |

`__post_init__` automatically derives sensible defaults (e.g., GDN head dimensions and
the hybrid layer schedule) if they are not explicitly set.

---

## Quick start

```bash
pip install -r requirements.txt
python -m qwen35
```

### Train a BPE tokenizer only
```bash
python train_tokenizer.py --data data/sample.txt --output runs/demo/tokenizer.json --vocab-size 512
```

### Train the model
```bash
python train.py --data data/sample.txt --output-dir runs/demo --max-steps 200 --vocab-size 512 --no-hybrid
```

### Reuse an existing tokenizer
```bash
python train.py --data data/sample.txt --output-dir runs/demo --tokenizer runs/demo/tokenizer.json
```

### Use FlashAttention / FLA (optional)
```bash
pip install flash-attn fla
python train.py --data data/sample.txt --output-dir runs/demo
```

### Distributed training (multi-GPU)
```bash
torchrun --nproc_per_node=2 train.py --data data/sample.txt --output-dir runs/ddp --distributed --max-steps 200
```

### Run inference from a checkpoint
```bash
python inference.py --checkpoint runs/demo/checkpoint_final.pt --prompt "Once upon a time" --max-new-tokens 128
```

### Load HuggingFace-format weights
```python
from qwen35 import Qwen35ForCausalLM
model = Qwen35ForCausalLM.from_pretrained_hf("path/to/hf/checkpoint")
```

### Useful CLI flags
- `--vocab-size` — target BPE vocabulary size (default 4096)
- `--no-hybrid` — use only full attention (faster on CPU)
- `--moe` — enable mixture-of-experts FFN
- `--num-experts` — number of MoE experts (default 8)
- `--eval-data` — optional held-out file for perplexity evaluation
- `--eval-batches` — cap validation batches for quicker checks
- `--mixed-precision {none,auto,bf16,fp16}` — CUDA autocast mode
- `--gradient-checkpointing` — reduce activation memory by recomputing
- `--distributed` — enable DistributedDataParallel (requires torchrun or similar)
- `--flash` — force FlashAttention/FLA kernels in inference when available

---

## File layout

```text
qwen35/
  config.py          # Qwen35Config dataclass and layer schedule builder
  cache.py           # ModelCache: KV cache + GDN recurrent/conv state
  rope.py            # Rotary embeddings (RoPE)
  norms.py           # RMSNorm and RMSNormGated
  utils.py           # repeat_kv (GQA) and l2norm (GDN kernel helper)
  attention.py       # Full softmax attention with GQA + FlashAttention dispatch
  gdn.py             # Gated DeltaNet: chunked + recurrent kernels + short conv
  mlp.py             # SwiGLU dense FFN and MoESwiGLU
  layers.py          # DecoderLayer, GDNLayer, and build_decoder_layer factory
  model.py           # TextModel, Qwen35ForCausalLM, save/load helpers
  generation.py      # Autoregressive generate() with cache
  kernels.py         # Optional FlashAttention / FLA dispatch
  training/
    bpe.py           # Byte-level BPE tokenizer (train, encode, decode, save)
    data.py          # TextLMDataset and corpus loader
    trainer.py       # Training loop, evaluation, mixed precision, DDP
train.py              # Main training entrypoint
train_tokenizer.py    # Tokenizer-only entrypoint
inference.py          # Standalone inference CLI
```

---

## Reference

GDN recurrence follows the torch fallbacks used in Hugging Face Qwen3-Next and OLMo
Hybrid-style implementations.
