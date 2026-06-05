"""Quick smoke test: python -m qwen35"""

import torch

from qwen35 import Qwen35Config, Qwen35ForCausalLM, generate
from qwen35.training.bpe import BPETokenizer


def main() -> None:
    torch.manual_seed(123)

    text = "Hello BPE world! " * 20
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=512)

    config = Qwen35Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=4,
        use_hybrid_attention=True,
        gdn_interval=4,
    )
    model = Qwen35ForCausalLM(config)

    input_ids = torch.tensor([tokenizer.encode(text[:80])], dtype=torch.long)
    out = model(input_ids, labels=input_ids)

    print("layer types:", config.layer_types)
    print("vocab_size:", tokenizer.vocab_size)
    print("parameters:", f"{model.num_parameters():,}")
    print("loss:", f"{out.loss.item():.4f}")
    print("decode sample:", repr(tokenizer.decode(input_ids[0, :16].tolist())))

    generated = generate(model, input_ids[:, :8], max_new_tokens=4, temperature=0.0)
    print("generated:", repr(tokenizer.decode(generated[0].tolist())))


if __name__ == "__main__":
    main()
