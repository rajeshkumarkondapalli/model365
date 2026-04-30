# model365 — a tiny GPT from scratch

A minimal, from-scratch decoder-only Transformer (the same architecture family as GPT/Claude), trainable on a Linux laptop.

This is **educational scale**, not frontier scale: a few-million-parameter character-level LM that learns to babble Shakespeare in a few minutes. The architecture (multi-head causal self-attention, MLP, LayerNorm, residual streams, weight tying, sampling with temperature/top-k) is the real thing — just smaller.

The server also exposes **image, music, and video generation** endpoints backed by pretrained open-source models (SD-Turbo, MusicGen, ModelScope T2V). The text model is the only thing here that's actually built from scratch — multimodal generators wrap existing checkpoints, since training those from zero requires frontier-scale compute.

## Files

| File | Purpose |
|---|---|
| `model.py` | GPT architecture (attention, MLP, blocks, sampler) |
| `data.py` | Downloads TinyShakespeare and builds char tokenizer |
| `train.py` | Training loop |
| `serve.py` | FastAPI inference endpoints |
| `image_gen.py` | Text-to-image (SD-Turbo via `diffusers`) |
| `music_gen.py` | Text-to-music (MusicGen-small via `transformers`) |
| `video_gen.py` | Text-to-video (ModelScope T2V via `diffusers`) |

## Setup (Linux laptop)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, the standard `torch` wheel will pick up CUDA automatically. CPU also works (slower).

## Train

```bash
# Build dataset (auto-downloads TinyShakespeare ~1MB)
python data.py

# Train (defaults: ~3M params, 2000 steps)
python train.py
```

On a modern CPU this is a few minutes; on a laptop GPU it's ~1 minute. You should see val loss drop from ~4.2 to ~1.6.

Tweak size / steps:

```bash
python train.py --steps 5000 --n_layer 6 --n_head 6 --n_embd 384 --block_size 256
```

A checkpoint is written to `ckpt.pt`.

## Serve

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Returns | Backed by |
|---|---|---|---|
| GET  | `/health`   | JSON         | — |
| POST | `/generate` | JSON (text)  | Your trained GPT (`ckpt.pt`) |
| POST | `/image`    | `image/png`  | `stabilityai/sd-turbo` |
| POST | `/music`    | `audio/wav`  | `facebook/musicgen-small` |
| POST | `/video`    | `video/mp4`  | `damo-vilab/text-to-video-ms-1.7b` |

Each generator is **lazy-loaded** on first request — the relevant Hugging Face checkpoint downloads on first use (a few GB) and stays in memory.

Open http://localhost:8000/docs for the auto-generated Swagger UI.

### Try it

Text:
```bash
curl -s http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"ROMEO:","max_new_tokens":200,"temperature":0.9,"top_k":40}'
```

Image:
```bash
curl -s -X POST http://localhost:8000/image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a watercolor of a fox in a forest","steps":2}' \
  --output out.png
```

Music:
```bash
curl -s -X POST http://localhost:8000/music \
  -H 'content-type: application/json' \
  -d '{"prompt":"lofi hip hop, mellow piano, rain","duration_seconds":5}' \
  --output out.wav
```

Video (GPU strongly recommended):
```bash
curl -s -X POST http://localhost:8000/video \
  -H 'content-type: application/json' \
  -d '{"prompt":"a panda dancing on the moon","num_frames":16}' \
  --output out.mp4
```

### Hardware notes

| Modality | Model size | CPU laptop | Laptop GPU (8GB) |
|---|---|---|---|
| Text (your GPT) | ~3M params | instant | instant |
| Image (SD-Turbo) | ~1.5GB | ~30–60s / image | ~1s / image |
| Music (MusicGen-small) | ~1.5GB | ~30–60s / 5s clip | ~5s / 5s clip |
| Video (ModelScope T2V) | ~3.5GB | impractical (10+ min) | ~30–60s / 2s clip |

Set `DEVICE=cuda` (or leave unset — auto-detects). Models download to `~/.cache/huggingface/`.

## Scaling up from here

This template is intentionally tiny. To make it more capable:

1. **Bigger tokenizer** — replace char-level with BPE (`sentencepiece` or `tiktoken`).
2. **Bigger data** — swap TinyShakespeare for FineWeb-Edu / OpenWebText.
3. **Bigger model** — increase `n_layer`, `n_head`, `n_embd`. Add RoPE, RMSNorm, SwiGLU (LLaMA-style).
4. **Faster** — use `torch.compile`, mixed precision (`bfloat16`), `F.scaled_dot_product_attention`, gradient accumulation.
5. **Instruction tuning** — fine-tune on SFT data (Alpaca, OpenAssistant) for chat behavior.
6. **Alignment** — DPO/RLHF on preference pairs.

Going from this to "ChatGPT-like" is a question of compute and data, not code complexity.
