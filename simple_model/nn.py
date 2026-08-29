"""A tiny neural network implemented from scratch with NumPy.

No autograd, no ML framework: every layer implements its own forward
and backward pass, and gradients are wired together manually in
`Sequential`.
"""

import numpy as np


class Dense:
    """Fully-connected layer: y = x @ W + b."""

    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng()
        limit = np.sqrt(6 / (n_in + n_out))
        self.W = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask

    def params_and_grads(self):
        return []


class Sigmoid:
    def __init__(self):
        self._out = None

    def forward(self, x):
        self._out = 1 / (1 + np.exp(-x))
        return self._out

    def backward(self, grad_out):
        return grad_out * self._out * (1 - self._out)

    def params_and_grads(self):
        return []


class Sequential:
    """Chains layers together, forwarding activations and backpropagating
    gradients in reverse order."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()


class MSELoss:
    def forward(self, pred, target):
        self._diff = pred - target
        return np.mean(self._diff ** 2)

    def backward(self):
        n = self._diff.size
        return 2 * self._diff / n


class SGD:
    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self):
        for param, grad in self.model.params_and_grads():
            param -= self.lr * grad
