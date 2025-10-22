import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from numba import njit

import bark.forest as forest
from bark import types
from bark.enums import FeatureTypeEnum, NodeState
from bark.utils.bit_operations import next_power_of_2


@njit
def pre_order_traverse(
    nodes: np.ndarray,
) -> list[int]:
    stack = []
    node_idxs = []
    current_idx = 0

    while True:
        node_idxs.append(current_idx)

        if not nodes[current_idx]["is_leaf"]:
            stack.append(forest.left(current_idx))
            stack.append(forest.right(current_idx))

        if not stack:
            return node_idxs
        current_idx = stack.pop(0)


@jax.jit
def terminal_nodes(
    feature_idx_tree: Int[Array, "... 2**max_depth"],
) -> Bool[Array, "... 2**max_depth"]:
    """Find all leaves"""
    return feature_idx_tree == NodeState.Leaf


@jax.jit
def singly_internal_nodes(
    feature_idx_tree: Int[Array, "... 2**max_depth"],
) -> Bool[Array, "... 2**max_depth"]:
    """Find all decision nodes where both children are leaves"""
    node_limit = feature_idx_tree.shape[-1]
    all_node_idcs = jnp.arange(node_limit)

    # we clip the left/right idcs to prevent accessing out of bounds
    # this is fine since the last node_limit//2 nodes must be leaves if active,
    # so will not be singly internal
    left_idcs = forest.left(all_node_idcs).clip(0, node_limit - 1)
    right_idcs = forest.right(all_node_idcs).clip(0, node_limit - 1)

    return (
        feature_idx_tree
        != NodeState.Inactive & feature_idx_tree
        != NodeState.Leaf & feature_idx_tree[left_idcs]
        == NodeState.Leaf & feature_idx_tree[right_idcs]
        == NodeState.Leaf
    )


@jax.jit
def get_node_subspace(
    feature_idx_tree: Int[Array, "... 2**max_depth"],
    threshold_tree: Float[Array, "m 2**max_depth"],
    node_idx: Int[Array, "..."],
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
):
    """Get the subset of the domain that reaches a given node."""
    subspace = bounds.copy()
    parent_idx = forest.parent(node_idx)
    while node_idx != 0:
        feature_idx = feature_idx_tree[parent_idx]

        if feat_types[feature_idx] == FeatureTypeEnum.Cat:
            if node_idx == forest.left(parent_idx):
                subspace[feature_idx, 1] = int(parent_node["threshold"]) & int(
                    subspace[feature_idx, 1]
                )
            else:
                max_threshold = next_power_of_2(int(subspace[feature_idx, 1])) - 1
                neg_threshold = max_threshold - parent_node["threshold"]
                subspace[feature_idx, 1] = int(neg_threshold) & int(
                    subspace[feature_idx, 1]
                )
        else:
            if node_idx == forest.left(parent_idx):
                subspace[feature_idx, 1] = min(
                    parent_node["threshold"], subspace[feature_idx, 1]
                )
            else:
                int_delta = (
                    1 if feat_types[feature_idx] == FeatureTypeEnum.Int.value else 0
                )
                subspace[feature_idx, 0] = max(
                    parent_node["threshold"] + int_delta, subspace[feature_idx, 0]
                )

        node_idx, parent_idx = parent_idx, forest.parent(parent_idx)

    return subspace
