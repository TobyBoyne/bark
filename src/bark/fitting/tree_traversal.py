import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

import bark.forest as forest
from bark import types
from bark.enums import FeatureTypeEnum, NodeState


@jax.jit
def terminal_nodes(
    feature_idx_tree: Int[Array, "... max_nodes"],
) -> Bool[Array, "... max_nodes"]:
    """Find all leaves"""
    return feature_idx_tree == NodeState.Leaf


@jax.jit
def singly_internal_nodes(
    feature_idx_tree: Int[Array, "... max_nodes"],
) -> Bool[Array, "... max_nodes"]:
    """Find all decision nodes where both children are leaves"""
    node_limit = feature_idx_tree.shape[-1]
    all_node_idcs = jnp.arange(node_limit)

    # we clip the left/right idcs to prevent accessing out of bounds
    # this is fine since the last node_limit//2 nodes must be leaves if active,
    # so will not be singly internal
    left_idcs = forest.left(all_node_idcs).clip(0, node_limit - 1)
    right_idcs = forest.right(all_node_idcs).clip(0, node_limit - 1)

    return (
        (feature_idx_tree != NodeState.Inactive)
        & (feature_idx_tree != NodeState.Leaf)
        & (feature_idx_tree[left_idcs] == NodeState.Leaf)
        & (feature_idx_tree[right_idcs] == NodeState.Leaf)
    )


def get_node_subspace(
    tree: types.Tree,
    node_idx: Int[Array, ""],
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
):
    """Get the subset of the domain that reaches a given node."""

    def cond(v: tuple[Int[Array, ""], types.BoundsT]):
        node_idx, _ = v
        return node_idx != 0

    def reduce_subspace(v: tuple[Int[Array, ""], types.BoundsT]):
        node_idx, subspace = v

        parent_idx = forest.parent(node_idx)
        feature_idx = tree.feature_idx[parent_idx]
        is_left = node_idx == forest.left(parent_idx)

        cat_threshold = tree.threshold[parent_idx].astype(jnp.uint64)
        cat_threshold = jnp.where(is_left, cat_threshold, ~cat_threshold)
        cat_threshold = cat_threshold & subspace[1, feature_idx].astype(jnp.uint64)
        cat_subspace = subspace.at[1, feature_idx].set(
            cat_threshold.astype(subspace.dtype)
        )

        ord_threshold = tree.threshold[parent_idx]
        # if the node is to the right of parent, then we start the integer bound
        # from `threshold + 1` to avoid intersection.
        int_delta = jnp.where(feat_types[feature_idx] == FeatureTypeEnum.Int, 1.0, 0.0)
        ord_threshold = jnp.stack(
            (
                jnp.where(is_left, subspace[0, feature_idx], ord_threshold + int_delta),
                jnp.where(is_left, ord_threshold, subspace[1, feature_idx]),
            ),
            axis=-1,
        )
        ord_subspace = subspace.at[:, feature_idx].set(ord_threshold)

        subspace = jnp.where(
            feat_types[feature_idx] == FeatureTypeEnum.Cat, cat_subspace, ord_subspace
        )
        return forest.parent(node_idx), subspace

    node_idx, subspace = jax.lax.while_loop(cond, reduce_subspace, (node_idx, bounds))
    return subspace
