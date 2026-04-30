"""FastAPI inference endpoint for the trained GPT.

Run: uvicorn serve:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import load_vocab
from model import GPT, GPTConfig

CKPT_PATH = os.environ.get("CKPT_PATH", "ckpt.pt")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

state = {}


def encode(s: str, stoi):
    # unknown chars dropped silently
    return [stoi[c] for c in s if c in stoi]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(
            f"Checkpoint not found at {CKPT_PATH}. Train first: python train.py"
        )
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    stoi, itos = load_vocab()
    state.update(model=model, cfg=cfg, stoi=stoi, itos=itos)
    print(f"loaded {CKPT_PATH} on {DEVICE}; vocab={len(stoi)}; block={cfg.block_size}")
    yield
    state.clear()


app = FastAPI(title="model365 inference", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="\n", description="Seed text")
    max_new_tokens: int = Field(default=200, ge=1, le=2000)
    temperature: float = Field(default=0.9, gt=0.0, le=5.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=1000)


class GenerateResponse(BaseModel):
    completion: str
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "loaded": "model" in state}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    model: GPT = state["model"]
    stoi, itos = state["stoi"], state["itos"]

    ids = encode(req.prompt, stoi)
    if not ids:
        ids = [stoi.get("\n", 0)]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(
        x,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    new_ids = out[0, len(ids):].tolist()
    return GenerateResponse(prompt=req.prompt, completion=decode(new_ids, itos))
