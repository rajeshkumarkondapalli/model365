"""Text-to-image via Stable Diffusion Turbo (pretrained, from Hugging Face).

Lazy-loaded singleton: the model is only downloaded/loaded on first request.
"""
import io
import os
import threading
from typing import Optional

import torch
from PIL import Image

MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "stabilityai/sd-turbo")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

_pipe = None
_lock = threading.Lock()


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        kwargs = {"torch_dtype": dtype}
        if DEVICE == "cuda":
            kwargs["variant"] = "fp16"
        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **kwargs)
        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
        return _pipe


def generate(
    prompt: str,
    steps: int = 1,
    guidance_scale: float = 0.0,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
) -> bytes:
    pipe = _get_pipe()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    image: Image.Image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
