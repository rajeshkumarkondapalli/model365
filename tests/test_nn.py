import numpy as np

from simple_model.train_xor import X, Y, train


def test_xor_learned():
    model = train(epochs=2000, lr=0.5, seed=0)
    preds = model.forward(X)
    predicted_labels = (preds > 0.5).astype(float)
    assert np.array_equal(predicted_labels, Y)


def test_loss_decreases():
    from simple_model.nn import MSELoss

    rng = np.random.default_rng(1)
    from simple_model.train_xor import build_model
    from simple_model.nn import SGD

    model = build_model(rng)
    loss_fn = MSELoss()
    optimizer = SGD(model, lr=0.5)

    pred = model.forward(X)
    first_loss = loss_fn.forward(pred, Y)
    model.backward(loss_fn.backward())
    optimizer.step()

    for _ in range(500):
        pred = model.forward(X)
        loss = loss_fn.forward(pred, Y)
        model.backward(loss_fn.backward())
        optimizer.step()

    assert loss < first_loss
