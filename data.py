"""Tiny char-level dataset. Downloads TinyShakespeare on first run."""
import os
import json
from pathlib import Path
import requests
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download():
    DATA_DIR.mkdir(exist_ok=True)
    raw = DATA_DIR / "input.txt"
    if not raw.exists():
        print(f"Downloading TinyShakespeare to {raw} ...")
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        raw.write_text(r.text, encoding="utf-8")
    return raw.read_text(encoding="utf-8")


def build():
    text = download()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(data))
    np.save(DATA_DIR / "train.npy", data[:n])
    np.save(DATA_DIR / "val.npy", data[n:])
    (DATA_DIR / "vocab.json").write_text(
        json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}})
    )
    print(f"vocab_size={len(chars)}  train={len(data[:n])}  val={len(data[n:])}")
    return len(chars)


def load_split(split: str):
    return np.load(DATA_DIR / f"{split}.npy")


def load_vocab():
    v = json.loads((DATA_DIR / "vocab.json").read_text())
    stoi = v["stoi"]
    itos = {int(k): val for k, val in v["itos"].items()}
    return stoi, itos


if __name__ == "__main__":
    build()
