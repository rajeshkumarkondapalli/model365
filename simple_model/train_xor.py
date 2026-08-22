"""Train the from-scratch network on XOR, the classic non-linearly
separable toy problem, to prove the model actually learns."""

import numpy as np

from simple_model.nn import Dense, MSELoss, ReLU, SGD, Sequential, Sigmoid

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def build_model(rng):
    return Sequential([
        Dense(2, 8, rng=rng),
        ReLU(),
        Dense(8, 1, rng=rng),
        Sigmoid(),
    ])


def train(epochs=2000, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    model = build_model(rng)
    loss_fn = MSELoss()
    optimizer = SGD(model, lr=lr)

    for epoch in range(1, epochs + 1):
        pred = model.forward(X)
        loss = loss_fn.forward(pred, Y)
        model.backward(loss_fn.backward())
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f"epoch {epoch:5d}  loss {loss:.4f}")

    return model


if __name__ == "__main__":
    model = train()
    preds = model.forward(X)
    print("\nfinal predictions:")
    for inputs, target, pred in zip(X, Y, preds):
        print(f"  {inputs.tolist()} -> {pred[0]:.3f} (target {target[0]:.0f})")
