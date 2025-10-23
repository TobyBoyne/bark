"""A collection of pre-built trees, used for testing."""

from typing import cast

import jax
import jax.numpy as jnp
from flax import struct

from bark import forest, types
from bark.enums import FeatureTypeEnum
from bark.fitting.tree_proposals import grow


@struct.dataclass
class TreesTestCase:
    trees: types.Trees
    bounds: types.BoundsT
    feat_types: types.FeatTypesT


def get_continuous_trees_test_case() -> TreesTestCase:
    m = 3
    trees = forest.create_empty_forest(m=m, max_depth=6)
    bounds = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]], dtype=trees.threshold.dtype
    )
    feat_types = jnp.full(bounds.shape[1], FeatureTypeEnum.Cont)
    grow_vmap = jax.vmap(grow, in_axes=0)
    trees = grow_vmap(
        cast(types.Tree, trees),
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
    return TreesTestCase(
        trees=cast(types.Trees, trees), bounds=bounds, feat_types=feat_types
    )
