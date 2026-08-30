"""Loads a trained checkpoint and continues a given text prompt,
demonstrating the model's input -> output behavior directly (as opposed
to `train.py`, which only samples from an empty context)."""

import argparse

import torch

from simple_llm.model import GPT
from simple_llm.tokenizer import CharTokenizer
from simple_llm.train import CHECKPOINT_PATH


def load_model(path=CHECKPOINT_PATH):
    checkpoint = torch.load(path, weights_only=True)
    tokenizer = CharTokenizer.from_vocab(checkpoint["stoi"], checkpoint["itos"])
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=checkpoint["block_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def continue_text(prompt, max_new_tokens=100, path=CHECKPOINT_PATH):
    model, tokenizer = load_model(path)
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    return tokenizer.decode(generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue a text prompt with the trained model.")
    parser.add_argument("prompt", nargs="?", default="the quick brown fox")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    output = continue_text(args.prompt, max_new_tokens=args.max_new_tokens)
    print(f"input:  {args.prompt!r}")
    print(f"output: {output!r}")
