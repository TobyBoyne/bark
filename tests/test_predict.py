import math

import jax
import jax.numpy as jnp

from bark.testing.bark_test_cases import get_continuous_bark_test_case
from bark.tree_kernels.tree_gps import forest_predict


def test_forest_predict():
    bark_test_case = get_continuous_bark_test_case()
    key = jax.random.key(81723)
    bounds = bark_test_case.data.bounds
    test_X = jax.random.uniform(
        key, (7, bounds.shape[1]), minval=bounds[0], maxval=bounds[1]
    )

    batch_shape = (4, 2)
    batched_bark_model = jax.tree_util.tree_map(
        lambda x: jnp.tile(x[None], (*batch_shape, *[1 for _ in x.shape])),
        bark_test_case.bark_model,
    )
    mu, var = forest_predict(batched_bark_model, bark_test_case.data, test_X, diag=True)

    batch_size = math.prod(batch_shape)
    assert mu.shape == (batch_size, test_X.shape[0])
    assert var.shape == (batch_size, test_X.shape[0])

    mu, var = forest_predict(
        batched_bark_model, bark_test_case.data, test_X, diag=False
    )

    assert mu.shape == (batch_size, test_X.shape[0])
    assert var.shape == (batch_size, test_X.shape[0], test_X.shape[0])
