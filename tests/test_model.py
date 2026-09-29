"""Small deterministic smoke test; the main run measures actual performance."""

import numpy as np

from thinfilm.model import predict, train_model


def test_training_is_reproducible_for_same_seed() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(40, 180, (32, 4))
    y = np.broadcast_to(((x[:, :1] - 40) / 140), (32, 41)).copy()
    a = train_model(x[:24], y[:24], x[24:], y[24:], low=40, high=180, seed=270069, epochs=3)
    b = train_model(x[:24], y[:24], x[24:], y[24:], low=40, high=180, seed=270069, epochs=3)
    np.testing.assert_array_equal(predict(a.model, x, 40, 180), predict(b.model, x, 40, 180))
    assert a.history == b.history
