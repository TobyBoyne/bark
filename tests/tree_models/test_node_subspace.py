import jax
import jax.numpy as jnp

from bark import types
from bark.enums import FeatureTypeEnum, NodeState
from bark.fitting.tree_traversal import get_node_subspace
from bark.forest import create_empty_forest

jax.config.update("jax_enable_x64", True)


def test_node_subspace():
    trees = create_empty_forest(m=3)
    trees = types.Trees(
        feature_idx=trees.feature_idx.at[:, :3].set(
            jnp.array(
                [0, NodeState.Leaf, NodeState.Leaf], dtype=trees.feature_idx.dtype
            )
        ),
        threshold=trees.threshold.at[:, 0].set(0.5),
    )

    bounds = jnp.zeros((2, 4), dtype=jnp.float32)
    bounds = bounds.at[1, :].set(1.0)
    feat_types = jnp.full((bounds.shape[1],), FeatureTypeEnum.Cont)
    node_idcs = jnp.array([0, 1, 2], dtype=jnp.uint64)

    subspace = jax.vmap(get_node_subspace, in_axes=(0, 0, 0, None, None))(
        trees.feature_idx,
        trees.threshold,
        node_idcs,
        bounds,
        feat_types,
    )

    assert subspace.shape[-2:] == bounds.shape
    assert (subspace[0] == bounds).all()
    assert (
        subspace[1] == jnp.array([[0.0, 0.0, 0.0, 0.0], [0.5, 1.0, 1.0, 1.0]])
    ).all()
    assert (
        subspace[2] == jnp.array([[0.5, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    ).all()
