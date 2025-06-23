import numpy as np
from numba import njit

import bark.forest as forest
from bark.forest import FeatureTypeEnum
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


@njit
def terminal_nodes(nodes: np.ndarray) -> list[int]:
    """Find all leaves"""
    terminal_idcs = []
    for node_idx in pre_order_traverse(nodes):
        node = nodes[node_idx]
        if node["is_leaf"]:
            terminal_idcs.append(node_idx)

    return terminal_idcs


@njit
def singly_internal_nodes(nodes: np.ndarray) -> list[int]:
    """Find all decision nodes where both children are leaves"""
    singly_internal_idcs = []
    for node_idx in pre_order_traverse(nodes):
        node = nodes[node_idx]
        if (
            (1 - node["is_leaf"])
            and nodes[forest.left(node_idx)]["is_leaf"]
            and nodes[forest.right(node_idx)]["is_leaf"]
        ):
            singly_internal_idcs.append(node_idx)

    return singly_internal_idcs


@njit
def get_node_subspace(
    tree: np.ndarray, node_idx: int, bounds: np.ndarray, feat_types: np.ndarray
):
    """Get the subset of the domain that reaches a given node."""
    subspace = bounds.copy()
    parent_idx = forest.parent(node_idx)
    while node_idx != 0:
        parent_node = tree[parent_idx]
        feature_idx = parent_node["feature_idx"]

        if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
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
