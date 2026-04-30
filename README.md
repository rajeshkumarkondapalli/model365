# model365 — a tiny GPT from scratch

A minimal, from-scratch decoder-only Transformer (the same architecture family as GPT/Claude), trainable on a Linux laptop.

This is **educational scale**, not frontier scale: a few-million-parameter character-level LM that learns to babble Shakespeare in a few minutes. The architecture (multi-head causal self-attention, MLP, LayerNorm, residual streams, weight tying, sampling with temperature/top-k) is the real thing — just smaller.

## Files

| File | Purpose |
|---|---|
| `model.py` | GPT architecture (attention, MLP, blocks, sampler) |
| `data.py` | Downloads TinyShakespeare and builds char tokenizer |
| `train.py` | Training loop |
| `serve.py` | FastAPI inference endpoint |

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

`GET /health` — readiness check.

`POST /generate` — body:

```json
{
  "prompt": "ROMEO:",
  "max_new_tokens": 200,
  "temperature": 0.9,
  "top_k": 40
}
```

### Try it

```bash
curl -s http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"ROMEO:","max_new_tokens":200,"temperature":0.9,"top_k":40}' | python -m json.tool
```

Open http://localhost:8000/docs for the auto-generated Swagger UI.

## Scaling up from here

This template is intentionally tiny. To make it more capable:

1. **Bigger tokenizer** — replace char-level with BPE (`sentencepiece` or `tiktoken`).
2. **Bigger data** — swap TinyShakespeare for FineWeb-Edu / OpenWebText.
3. **Bigger model** — increase `n_layer`, `n_head`, `n_embd`. Add RoPE, RMSNorm, SwiGLU (LLaMA-style).
4. **Faster** — use `torch.compile`, mixed precision (`bfloat16`), `F.scaled_dot_product_attention`, gradient accumulation.
5. **Instruction tuning** — fine-tune on SFT data (Alpaca, OpenAssistant) for chat behavior.
6. **Alignment** — DPO/RLHF on preference pairs.

Going from this to "ChatGPT-like" is a question of compute and data, not code complexity.
