"""Inspired by https://github.com/ogrisel/pygbm and https://github.com/Gattocrucco/bartz"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, UInt

from bark import enums, types
from bark.utils.bit_operations import next_power_of_2_exponent


def is_leaf(feature_idx: types.IndexT) -> Bool[Array, "..."] | bool:
    """Return a mask for all leaves in a tree."""
    return feature_idx == -1


def depth(idx: types.IndexT) -> types.IndexT:
    """Get the depth of node at `idx` in the binary tree."""
    return next_power_of_2_exponent(idx + 1) - 1


def parent(idx: types.IndexT) -> types.IndexT:
    """Get the parent of node at `idx` in the binary tree."""
    return (idx - 1) // 2


def left(idx: types.IndexT) -> types.IndexT:
    """Get the left child of node at `idx` in the binary tree."""
    return 2 * idx + 1


def right(idx: types.IndexT) -> types.IndexT:
    """Get the right child of node at `idx` in the binary tree."""
    return 2 * idx + 2


def _pass_one_through_tree(
    X: Float[Array, " d"],
    tree: types.Tree,
    feat_types: types.FeatTypesT,
) -> UInt[Array, ""]:
    # https://github.com/Gattocrucco/bartz/blob/a9607515a328a7e74cf39d9c396e3c1f072c8c93/src/bartz/grove.py#L108
    carry = (
        jnp.zeros((), bool),
        jnp.zeros((), jnp.int64),
    )

    def loop(carry, _):
        leaf_found, index = carry

        feature_idx = tree.feature_idx[index]
        threshold = tree.threshold[index]

        is_cat = feat_types[feature_idx] == enums.FeatureTypeEnum.Cat.value

        leaf_found |= is_leaf(feature_idx)
        child_index = left(index)
        child_index += is_cat * (
            1 - ((1 << X[feature_idx].astype(jnp.int64)) & threshold.astype(jnp.int64))
        )
        child_index += (1 - is_cat) * (X[feature_idx] > threshold)
        index = jnp.where(leaf_found, index, child_index)

        return (leaf_found, index), None

    (_, index), _ = jax.lax.scan(
        loop, carry, None, enums.MAX_DEPTH, unroll=enums.MAX_DEPTH
    )
    return index  # pyright: ignore


pass_through_tree = jax.vmap(_pass_one_through_tree, (0, None, None))
pass_through_forest = jax.vmap(pass_through_tree, (None, 0, None), out_axes=1)


def get_leaf_vectors(
    X: Float[Array, "N d"],
    tree: types.Tree,
    feat_types: types.FeatTypesT,
) -> UInt[Array, "N"]:
    # NOTE: this function is not jitted since jnp.unique returns a dynamically shaped
    # array. It may be possible to do some kind of dispatch, with a static argnum,
    # however it would require a non-jitted function at some point.
    x_leaves = pass_through_tree(X, tree, feat_types)
    all_leaves = jnp.unique(x_leaves)
    leaf_vector = (jnp.equal(x_leaves[:, None], all_leaves[None, :])).astype(
        jnp.float64
    )
    return leaf_vector


def forest_gram_matrix(
    X1: Float[Array, "N d"],
    X2: Float[Array, "M d"],
    trees: types.Tree,
    feat_types: types.FeatTypesT,
) -> Float[Array, "N M"]:
    x1_leaves = pass_through_forest(X1, trees, feat_types)
    x2_leaves = pass_through_forest(X2, trees, feat_types)
    sim_mat = jnp.equal(x1_leaves[:, None, :], x2_leaves[None, :, :])  # N x M x m
    sim_mat = jnp.mean(sim_mat, axis=-1)
    return sim_mat


batched_forest_gram_matrix = jax.vmap(
    forest_gram_matrix, in_axes=(None, None, 0, 0, None)
)


# def batched_forest_gram_matrix_no_null(
#     X1: Float[Array, "N d"],
#     X2: Float[Array, "M d"],
#     feature_idx_trees: Int[Array, "batch m 2**max_depth"],
#     threshold_trees: Float[Array, "batch m 2**max_depth"],
#     feat_types: types.FeatTypesT,
# ) -> Float[Array, "N M"]:
#     """Compute the gram matrix after removing empty trees."""
#     sim_mat = batched_forest_gram_matrix(
#         X1, X2, feature_idx_trees, threshold_trees, feat_types
#     )

#     num_trees = feature_idx_trees.shape[-2]
#     roots = feature_idx_trees[:, :, 0]
#     num_null_trees = jnp.sum(is_leaf(roots), axis=-1)[:, None, None]
#     num_non_null_trees = num_trees - num_null_trees

#     scale = feature_idx_trees.shape[-2] / jnp.maximum(num_non_null_trees, 1)
#     return (sim_mat - num_null_trees / num_trees) * scale


def create_empty_forest(m: int, max_depth: int = 6) -> types.Tree:
    feature_idx = jnp.full(
        (m, 2**max_depth - 1), enums.NodeState.Inactive, dtype=jnp.int32
    )
    feature_idx = feature_idx.at[:, 0].set(enums.NodeState.Leaf)
    threshold = jnp.zeros((m, 2**max_depth - 1), dtype=jnp.float32)
    return types.Tree(
        feature_idx=feature_idx,
        threshold=threshold,
    )
