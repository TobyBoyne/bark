import jax
import jax.numpy as jnp

from bark import forest
from bark.enums import FeatureTypeEnum
from bark.testing.build_test_trees import build_trees_from_dict

jax.config.update("jax_enable_x64", True)

FEATURE_TYPES = jnp.array(
    [FeatureTypeEnum.Cont, FeatureTypeEnum.Int, FeatureTypeEnum.Cat]
)

tree_dicts = [
    {(0, 0.5): ({(2, 13): (None, None)}, None)},
    {(2, 2): (None, {(1, 4): (None, None)})},
]


def test_pass_through_tree():
    trees = build_trees_from_dict(tree_dicts)
    X = jnp.array(
        [
            [0.25, 2, 0],
            [0.75, 8, 1],
        ],
        dtype=jnp.float64,
    )

    with jax.disable_jit():
        leaves = forest.pass_through_forest(
            X,
            trees,
            FEATURE_TYPES,
        )
    assert jnp.all(leaves == jnp.array([[3, 5], [2, 1]], dtype=jnp.int64))
