"""Text-to-music via MusicGen-small (pretrained, from Hugging Face).

Output is a 32 kHz mono WAV byte string. ~50 audio tokens per second of music.
"""
import io
import os
import threading

import numpy as np
import soundfile as sf
import torch

MODEL_ID = os.environ.get("MUSIC_MODEL_ID", "facebook/musicgen-small")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
TOKENS_PER_SECOND = 50  # MusicGen audio codec frame rate

_state = {}
_lock = threading.Lock()


def _load():
    global _state
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()
        sr = model.config.audio_encoder.sampling_rate
        _state.update(processor=processor, model=model, sample_rate=sr)
        return _state


def generate(prompt: str, duration_seconds: float = 5.0) -> bytes:
    s = _load()
    processor = s["processor"]
    model = s["model"]
    sr = s["sample_rate"]

    max_new_tokens = max(16, int(duration_seconds * TOKENS_PER_SECOND))
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

    wav = audio[0, 0].detach().cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
