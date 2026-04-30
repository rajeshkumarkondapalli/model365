"""FastAPI inference server for model365.

Endpoints:
  GET  /health     - readiness
  POST /generate   - text completion from the from-scratch GPT
  POST /image      - text-to-image (SD-Turbo)
  POST /music      - text-to-music (MusicGen-small)
  POST /video      - text-to-video (ModelScope T2V)

Run: uvicorn serve:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

from data import load_vocab
from model import GPT, GPTConfig

CKPT_PATH = os.environ.get("CKPT_PATH", "ckpt.pt")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

state = {}


def encode(s: str, stoi):
    return [stoi[c] for c in s if c in stoi]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Text model is optional — server still serves /image, /music, /video without it.
    if os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
            cfg = GPTConfig(**ckpt["config"])
            model = GPT(cfg).to(DEVICE)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            stoi, itos = load_vocab()
            state.update(model=model, cfg=cfg, stoi=stoi, itos=itos)
            print(f"text model loaded from {CKPT_PATH} on {DEVICE}; vocab={len(stoi)}")
        except Exception as e:
            print(f"warning: failed to load text checkpoint ({e}); /generate disabled")
    else:
        print(f"no text checkpoint at {CKPT_PATH}; /generate disabled")
    yield
    state.clear()


app = FastAPI(title="model365 inference", lifespan=lifespan)


# ----- text -----
class GenerateRequest(BaseModel):
    prompt: str = Field(default="\n")
    max_new_tokens: int = Field(default=200, ge=1, le=2000)
    temperature: float = Field(default=0.9, gt=0.0, le=5.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=1000)


class GenerateResponse(BaseModel):
    completion: str
    prompt: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "text_model_loaded": "model" in state,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        raise HTTPException(503, "text model not loaded; train with python train.py")
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


# ----- image -----
class ImageRequest(BaseModel):
    prompt: str
    steps: int = Field(default=1, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    width: int = Field(default=512, ge=64, le=1024)
    height: int = Field(default=512, ge=64, le=1024)
    seed: Optional[int] = None


@app.post("/image")
async def image(req: ImageRequest):
    import image_gen
    try:
        png = await run_in_threadpool(
            image_gen.generate,
            prompt=req.prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(500, f"image generation failed: {e}")
    return Response(content=png, media_type="image/png")


# ----- music -----
class MusicRequest(BaseModel):
    prompt: str
    duration_seconds: float = Field(default=5.0, gt=0.0, le=30.0)


@app.post("/music")
async def music(req: MusicRequest):
    import music_gen
    try:
        wav = await run_in_threadpool(
            music_gen.generate,
            prompt=req.prompt,
            duration_seconds=req.duration_seconds,
        )
    except Exception as e:
        raise HTTPException(500, f"music generation failed: {e}")
    return Response(content=wav, media_type="audio/wav")


# ----- video -----
class VideoRequest(BaseModel):
    prompt: str
    num_frames: int = Field(default=16, ge=8, le=64)
    num_inference_steps: int = Field(default=25, ge=5, le=100)
    fps: int = Field(default=8, ge=1, le=30)
    seed: Optional[int] = None


@app.post("/video")
async def video(req: VideoRequest):
    import video_gen
    try:
        mp4 = await run_in_threadpool(
            video_gen.generate,
            prompt=req.prompt,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            fps=req.fps,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(500, f"video generation failed: {e}")
    return Response(content=mp4, media_type="video/mp4")
