"""Pretrains the from-scratch GPT on a small text corpus via next-token
prediction, then generates text to show the model actually learned
something about the corpus's character-level structure."""

from pathlib import Path

import torch

from simple_llm.model import GPT
from simple_llm.tokenizer import CharTokenizer

DATA_PATH = Path(__file__).parent / "data" / "sample.txt"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.pt"
BLOCK_SIZE = 32
BATCH_SIZE = 16


def load_corpus(repeats=20):
    return DATA_PATH.read_text() * repeats


def get_batch(data, block_size, batch_size, generator):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def train(text=None, steps=2000, lr=3e-3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    text = text if text is not None else load_corpus()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = GPT(vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE, generator)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    return model, tokenizer


def save_checkpoint(model, tokenizer, path=CHECKPOINT_PATH):
    torch.save(
        {
            "model_state": model.state_dict(),
            "block_size": model.block_size,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        path,
    )


if __name__ == "__main__":
    model, tokenizer = train()
    save_checkpoint(model, tokenizer)

    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=300)[0].tolist()
    print("\ngenerated text:\n")
    print(tokenizer.decode(generated))
