# setup.md — Deployment TODO

Cloud deployment plan for `model365`. Items are ordered by recommended sequence; check off as completed.

## Overview

| Goal | Target platform | Est. cost |
|---|---|---|
| Train the model once | RunPod / Vast.ai spot GPU | < $1 one-shot |
| Serve 24/7 (low traffic, CPU) | Fly.io or Render | ~$3–5 / month |
| Serve with GPU, scale-to-zero | Modal or RunPod Serverless | ~$0 idle, pay per call |
| Free public demo | Hugging Face Spaces (CPU) | Free |

---

## TODO

### 1. Train on a rented GPU (one-shot)

- [ ] Create a **RunPod** account (or Vast.ai / Lambda Labs)
- [ ] Launch a Pod: RTX 4090 or A6000, PyTorch image, ~$0.30–$0.70/hr
- [ ] SSH into the pod and run:
  ```bash
  git clone <this-repo>
  cd model365
  pip install -r requirements.txt
  python data.py
  python train.py --steps 5000 --n_layer 6 --n_head 6 --n_embd 384 --block_size 256
  ```
- [ ] Download `ckpt.pt` locally (e.g. `scp` or RunPod web file browser)
- [ ] **Terminate the pod** to stop billing

### 2. Containerize for deployment

- [ ] Add a `Dockerfile` at repo root:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  ENV CKPT_PATH=/app/ckpt.pt
  CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
  ```
- [ ] Add `.dockerignore` (exclude `.venv/`, `data/`, `__pycache__/`)
- [ ] Build & smoke-test locally:
  ```bash
  docker build -t model365 .
  docker run -p 8080:8080 -v $PWD/ckpt.pt:/app/ckpt.pt model365
  curl -s localhost:8080/health
  ```

### 3. Deploy: Fly.io (CPU, cheapest persistent option)

- [ ] Install `flyctl` and `fly auth login`
- [ ] `fly launch --no-deploy` (generates `fly.toml`; choose region near users)
- [ ] In `fly.toml` set:
  - `[[vm]] size = "shared-cpu-1x"` and `memory = "1gb"`
  - `[http_service] auto_stop_machines = true`, `min_machines_running = 0` (scale-to-zero)
- [ ] Upload checkpoint as a Fly volume or bake into image:
  ```bash
  fly volumes create ckpt --size 1
  ```
  *(or just `COPY ckpt.pt /app/ckpt.pt` in the Dockerfile if it's small)*
- [ ] `fly deploy`
- [ ] `fly status` and test `curl https://<app>.fly.dev/generate ...`

### 4. Deploy: Modal (GPU, scale-to-zero) — alternative

- [ ] `pip install modal && modal token new`
- [ ] Add `modal_app.py`:
  ```python
  import modal
  app = modal.App("model365")
  image = (modal.Image.debian_slim()
           .pip_install_from_requirements("requirements.txt")
           .add_local_dir(".", "/app"))

  @app.function(image=image, gpu="T4", scaledown_window=60)
  @modal.asgi_app()
  def fastapi_app():
      import sys; sys.path.insert(0, "/app")
      from serve import app as fastapi
      return fastapi
  ```
- [ ] Upload checkpoint via Modal Volume:
  ```bash
  modal volume create model365-ckpt
  modal volume put model365-ckpt ckpt.pt /ckpt.pt
  ```
- [ ] Mount the volume in the function and set `CKPT_PATH=/vol/ckpt.pt`
- [ ] `modal deploy modal_app.py` → returns public HTTPS URL

### 5. Deploy: Hugging Face Spaces (free public demo) — alternative

- [ ] Create a new Space, SDK = **Docker**, hardware = **CPU basic (free)**
- [ ] Push this repo to the Space's git remote
- [ ] Commit `ckpt.pt` via Git LFS (`git lfs track "*.pt"`)
- [ ] Space auto-builds and exposes the FastAPI endpoint at `https://<user>-<space>.hf.space`

### 6. Production hardening (do once it's actually used)

- [ ] Add API key auth to `/generate` (FastAPI `Depends` + header check)
- [ ] Add request rate limiting (`slowapi`)
- [ ] Add Prometheus `/metrics` (`prometheus-fastapi-instrumentator`)
- [ ] Set up uptime monitoring (UptimeRobot free, or Better Stack)
- [ ] Add structured logging (request id, prompt length, latency, tokens generated)
- [ ] Pin model version in checkpoint filename (`ckpt-v1.pt`) for safe rollouts

### 7. Cost guardrails

- [ ] Set a hard billing alert in the cloud provider (e.g. RunPod auto-stop, Fly spending limit, Modal usage cap)
- [ ] Confirm scale-to-zero is actually working (`fly status` should show 0 running machines when idle)
- [ ] Cap `max_new_tokens` server-side (already capped at 2000 in `serve.py`)

---

## Decision shortcut

- **Just want it running cheaply, today?** → Fly.io CPU (steps 2 + 3).
- **Need GPU but only when called?** → Modal (steps 2 + 4).
- **Want a free shareable URL?** → Hugging Face Spaces (step 5).
