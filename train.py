"""Train the GPT on TinyShakespeare. CPU works (slow); GPU is much faster."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data import build, load_split, load_vocab
from model import GPT, GPTConfig


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, iters=50):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--out", type=str, default="ckpt.pt")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if not (Path("data") / "train.npy").exists():
        build()
    stoi, _ = load_vocab()
    vocab_size = len(stoi)
    train_data = load_split("train")
    val_data = load_split("val")

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M  vocab_size={vocab_size}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % args.eval_every == 0 or step == 1:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | {elapsed:.1f}s")

    torch.save(
        {"model_state": model.state_dict(), "config": cfg.__dict__},
        args.out,
    )
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
