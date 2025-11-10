"""A collection of pre-built bark models, used for testing."""

import jax
import jax.numpy as jnp
from flax import struct

from bark import forest, types
from bark.enums import FeatureTypeEnum
from bark.fitting.tree_proposals import grow


@struct.dataclass
class BarkModelTestCase:
    bark_model: types.BARKModel
    data: types.Data


def get_continuous_bark_test_case() -> BarkModelTestCase:
    m = 3
    trees = forest.create_empty_forest(m=m, max_depth=6)
    bounds = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]], dtype=trees.threshold.dtype
    )
    feat_types = jnp.full(bounds.shape[1], FeatureTypeEnum.Cont)
    grow_vmap = jax.vmap(grow, in_axes=0)
    trees = grow_vmap(
        trees,
        node_idx=jnp.zeros(m, dtype=jnp.int32),
        new_feature_idx=jnp.arange(m, dtype=jnp.int32),
        new_threshold=jnp.array([0.5, 2.5, 1.0]),
    )
    trees = grow_vmap(
        trees,
        node_idx=jnp.array([1, 1, 2], dtype=jnp.int32),
        new_feature_idx=jnp.array([0, 2, 3], dtype=jnp.int32),
        new_threshold=jnp.array([0.25, 2.7, 4.0]),
    )

    bark_model = types.BARKModel(trees=trees, noise_var=jnp.array(1.0))

    key = jax.random.key(0)
    train_X = jax.random.uniform(
        key, (10, bounds.shape[1]), minval=bounds[0], maxval=bounds[1]
    )
    train_Y = train_X.sum(axis=-1, keepdims=True)
    data = types.Data(
        train_X=train_X, train_Y=train_Y, bounds=bounds, feat_types=feat_types
    )

    return BarkModelTestCase(
        bark_model=bark_model,
        data=data,
    )
