"""Text-to-video via ModelScope T2V (pretrained, from Hugging Face).

NOTE: practical only on a GPU. CPU inference for 16 frames takes many minutes.
Output is an MP4 byte string.
"""
import io
import os
import threading
from typing import Optional

import imageio
import numpy as np
import torch

MODEL_ID = os.environ.get("VIDEO_MODEL_ID", "damo-vilab/text-to-video-ms-1.7b")
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
        from diffusers import DiffusionPipeline

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        kwargs = {"torch_dtype": dtype}
        if DEVICE == "cuda":
            kwargs["variant"] = "fp16"
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **kwargs)
        pipe.to(DEVICE)
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload()
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe
        return _pipe


def generate(
    prompt: str,
    num_frames: int = 16,
    num_inference_steps: int = 25,
    fps: int = 8,
    seed: Optional[int] = None,
) -> bytes:
    pipe = _get_pipe()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = pipe(
        prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    # diffusers returns frames as List[np.ndarray] of shape (H, W, 3) uint8,
    # or (1, F, H, W, 3) depending on version. Normalize to a list of HxWx3 uint8.
    frames = result.frames
    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        frames = list(frames)
    elif isinstance(frames, list) and frames and isinstance(frames[0], list):
        frames = frames[0]

    frames = [np.asarray(f) for f in frames]
    if frames[0].dtype != np.uint8:
        frames = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]

    buf = io.BytesIO()
    with imageio.get_writer(buf, format="ffmpeg", mode="I", fps=fps, codec="libx264", quality=8) as writer:
        for f in frames:
            writer.append_data(f)
    return buf.getvalue()
