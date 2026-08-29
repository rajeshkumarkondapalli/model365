# model365

A simple neural network built from scratch with NumPy — no ML frameworks.
Every layer (`Dense`, `ReLU`, `Sigmoid`) implements its own forward and
backward pass, and gradients are wired together manually.

## Structure

- `simple_model/nn.py` — the network: layers, a `Sequential` container,
  `MSELoss`, and an `SGD` optimizer.
- `simple_model/train_xor.py` — trains the network on the XOR problem
  (a classic test since it isn't linearly separable).
- `tests/test_nn.py` — checks the model actually learns XOR and that
  loss decreases during training.

## Usage

```bash
pip install -r requirements.txt
python -m simple_model.train_xor
pytest
```