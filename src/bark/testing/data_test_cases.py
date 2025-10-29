import jax
import jax.numpy as jnp

from bark import types
from bark.enums import FeatureTypeEnum


def get_continuous_data_trid(N: int, dim: int):
    key_x, key_y = jax.random.split(jax.random.key(0))
    train_X = jax.random.uniform(key_x, (N, dim), dtype=jnp.float64)
    train_f = jnp.pow(train_X - 1, 2).sum(axis=-1, keepdims=True) - (
        train_X[:, :-1] * train_X[:, 1:]
    ).sum(axis=-1, keepdims=True)
    train_Y = (train_f - train_f.mean()) / train_f.std() + 0.1 * jax.random.normal(
        key_y, train_f.shape
    )
    return types.Data(
        train_X=train_X,
        train_Y=train_Y,
        bounds=jnp.zeros((2, dim)).at[1, :].set(1.0),
        feat_types=jnp.full((dim,), FeatureTypeEnum.Cont.value),
    )
