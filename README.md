# model365

A minimal GPT-style Large Language Model built from scratch with
PyTorch tensors and autograd — no `nn.Transformer`, `nn.MultiheadAttention`,
or pretrained weights. Every mechanism is implemented explicitly.

## Structure

- `simple_llm/tokenizer.py` — `CharTokenizer`, a character-level tokenizer
  (text <-> token ids).
- `simple_llm/model.py` — the model:
  - `SelfAttentionHead` — manual Query/Key/Value projections and causal
    scaled dot-product attention.
  - `MultiHeadAttention` — several attention heads run in parallel and
    combined.
  - `FeedForward` — the per-token MLP applied after attention.
  - `TransformerBlock` — attention + feed-forward, each in a residual
    connection with pre-layer-norm.
  - `GPT` — token embedding + positional embedding, stacked transformer
    blocks, final layer norm, and an output head producing next-token
    probabilities; `generate()` samples tokens autoregressively.
- `simple_llm/train.py` — trains the model via next-token prediction
  (cross-entropy loss) on `simple_llm/data/sample.txt`, then generates text.
- `tests/test_llm.py` — checks tokenizer round-tripping, forward-pass
  shapes, generation length, that loss decreases, and that a full training
  run produces valid output.

## How this maps to an LLM's computational workflow

```
Prompt -> Tokenization -> Token IDs -> Embeddings -> Positional Information
   -> Transformer Blocks -> Self-Attention -> Feed-Forward Processing
   -> Output Probabilities -> Token Selection -> Generated Response
```

| Concept | Where it lives |
|---|---|
| Input text / Tokenization | `CharTokenizer` |
| Token embedding | `GPT.token_embedding` |
| Positional information | `GPT.position_embedding` |
| Self-attention, Query/Key/Value | `SelfAttentionHead` |
| Multi-head attention | `MultiHeadAttention` |
| Feed-forward network | `FeedForward` |
| Transformer blocks (residual + norm) | `TransformerBlock` |
| Next-token prediction | `GPT.forward` (cross-entropy over vocab logits) |
| Pretraining | `simple_llm/train.py` (next-token prediction on a text corpus) |
| Output generation | `GPT.generate` (autoregressive sampling) |

This is a toy, character-level model meant to demonstrate every
architectural piece, not a production LLM — it has no instruction
tuning, RLHF, or retrieval augmentation.

## Usage

```bash
pip install -r requirements.txt
python -m simple_llm.train
pytest
```