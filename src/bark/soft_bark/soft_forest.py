import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from bark import enums, types

smooth_indicator = jax.lax.logistic


@jax.jit
def _pass_one_through_soft_tree(
    X: Float[Array, " d"],
    tree: types.SoftTree,
    feat_types: types.FeatTypesT,
) -> Float[Array, " max_nodes"]:
    # at every node, compute the splits
    cont_splits = smooth_indicator((X[tree.feature_idx] - tree.threshold) / tree.tau)
    cum_indicator = cont_splits
    start_depths = 2 ** np.arange(enums.MAX_DEPTH + 1) - 1

    # propagate the splits down the tree
    # jax.lax.scan doesn't support dynamic indexing, whereas with
    # a for loop, jax understands that d is static.
    for d in range(enums.MAX_DEPTH - 1):
        nodes_at_depth = cum_indicator[start_depths[d] : start_depths[d + 1]]
        cum_indicator = cum_indicator.at[
            start_depths[d + 1] : start_depths[d + 2] : 2
        ].mul(nodes_at_depth)
        cum_indicator = cum_indicator.at[
            start_depths[d + 1] + 1 : start_depths[d + 2] : 2
        ].mul(1 - nodes_at_depth)

    cum_indicator = jnp.where(
        tree.feature_idx == enums.NodeState.Inactive, 0.0, cum_indicator
    )
    return cum_indicator


pass_through_soft_tree = jax.vmap(_pass_one_through_soft_tree, (0, None, None))
pass_through_soft_forest = jax.vmap(pass_through_soft_tree, (None, 0, None), out_axes=1)


@jax.jit
def similarity_matrix(
    T1: Float[Array, "N m max_nodes"], T2: Float[Array, "M m max_nodes"]
) -> Float[Array, "N M"]:
    return jnp.vecdot(T1[:, None, ...], T2[None, :, ...], axis=-1).mean(axis=-1)


def soft_forest_covar_matrix(
    X1: Float[Array, "N d"],
    X2: Float[Array, "M d"],
    trees: types.Tree,
    feat_types: types.FeatTypesT,
) -> Float[Array, "N M"]:
    x1_leaves = pass_through_soft_forest(X1, trees, feat_types)
    x2_leaves = pass_through_soft_forest(X2, trees, feat_types)
    return similarity_matrix(x1_leaves, x2_leaves)


def soft_forest_gram_matrix(
    X1: Float[Array, "N d"],
    trees: types.Tree,
    feat_types: types.FeatTypesT,
) -> Float[Array, "N N"]:
    x1_leaves = pass_through_soft_forest(X1, trees, feat_types)
    return similarity_matrix(x1_leaves, x1_leaves)


def create_empty_soft_forest(m: int, max_depth: int = enums.MAX_DEPTH) -> types.Tree:
    feature_idx = jnp.full(
        (m, 2**max_depth - 1), enums.NodeState.Inactive, dtype=jnp.int32
    )
    feature_idx = feature_idx.at[:, 0].set(enums.NodeState.Leaf)
    threshold = jnp.zeros((m, 2**max_depth - 1), dtype=jnp.float32)
    tau = jnp.full((m,), 0.1)
    return types.SoftTree(
        feature_idx=feature_idx,
        threshold=threshold,
        tau=tau,
    )
