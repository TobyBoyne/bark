"""Inspired by https://github.com/ogrisel/pygbm and https://github.com/Gattocrucco/bartz"""

from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, UInt

from bark import enums, types
from bark.utils.bit_operations import next_power_of_2_exponent

IT = TypeVar("IT", int, Array, types.IndexT)


def is_leaf(feature_idx: IT) -> IT | bool:
    """Return a mask for all leaves in a tree."""
    return feature_idx == enums.NodeState.Leaf


def depth(idx: IT) -> IT:
    """Get the depth of node at `idx` in the binary tree."""
    return next_power_of_2_exponent(idx + 1) - 1


def parent(idx: IT) -> IT:
    """Get the parent of node at `idx` in the binary tree."""
    return (idx - 1) // 2


def left(idx: IT) -> IT:
    """Get the left child of node at `idx` in the binary tree."""
    return 2 * idx + 1


def right(idx: IT) -> IT:
    """Get the right child of node at `idx` in the binary tree."""
    return 2 * idx + 2


@jax.jit
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
    return index


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


@jax.jit
def similarity_matrix(
    T1: Float[Array, "N m"], T2: Float[Array, "M m"]
) -> Float[Array, "N M m"]:
    return jnp.equal(T1[:, None, :], T2[None, :, :])  # N x M x m


@jax.jit
def similarity_matrix_reduce(
    T1: Float[Array, "N m"], T2: Float[Array, "M m"]
) -> Float[Array, "N M"]:
    """Compute the proportion of values for which the pairwise leaves are equal.

    The similarity matrix G (N, M) is given by:
        G_{ij} = sum_k T1_{ik} == T2_{jk}
    """

    # Use a scan to avoid creating an N x M x m matrix
    # This should be possible with a nested vmap to automatically reduce
    # https://github.com/jax-ml/jax/discussions/9505
    def scan_func(carry, x):
        t1, t2 = x
        return carry + jnp.equal(t1[:, None], t2[None, :]), None

    carry = jnp.zeros((T1.shape[0], T2.shape[0]))
    # iterate over the trees, adding ones where trees agree
    carry, _ = jax.lax.scan(scan_func, carry, (T1.T, T2.T))
    return carry / T1.shape[1]


def forest_covar_matrix(
    X1: Float[Array, "N d"],
    X2: Float[Array, "M d"],
    trees: types.Tree,
    feat_types: types.FeatTypesT,
) -> Float[Array, "N M"]:
    x1_leaves = pass_through_forest(X1, trees, feat_types)
    x2_leaves = pass_through_forest(X2, trees, feat_types)
    return similarity_matrix_reduce(x1_leaves, x2_leaves)


def forest_gram_matrix(
    X1: Float[Array, "N d"],
    trees: types.Tree,
    feat_types: types.FeatTypesT,
) -> Float[Array, "N N"]:
    x1_leaves = pass_through_forest(X1, trees, feat_types)
    return similarity_matrix_reduce(x1_leaves, x1_leaves)


def forest_gram_matrix_no_null(
    X1: Float[Array, "N d"],
    trees: types.Tree,
    feat_types: types.FeatTypesT,
) -> Float[Array, "N N"]:
    """Compute the gram matrix after removing empty trees."""
    sim_mat = forest_gram_matrix(X1, trees, feat_types)

    num_trees = trees.feature_idx.shape[-2]
    roots = trees.feature_idx[..., 0]
    num_null_trees = jnp.sum(is_leaf(roots), axis=-1)
    num_non_null_trees = num_trees - num_null_trees

    scale = num_trees / jnp.maximum(num_non_null_trees, 1)
    return (sim_mat - num_null_trees / num_trees) * scale


def create_empty_forest(m: int, max_depth: int = enums.MAX_DEPTH) -> types.Tree:
    feature_idx = jnp.full(
        (m, 2**max_depth - 1), enums.NodeState.Inactive, dtype=jnp.int32
    )
    feature_idx = feature_idx.at[:, 0].set(enums.NodeState.Leaf)
    threshold = jnp.zeros((m, 2**max_depth - 1), dtype=jnp.float32)
    return types.Tree(
        feature_idx=feature_idx,
        threshold=threshold,
    )
