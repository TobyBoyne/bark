import numpy as np
from numba import njit

from bark.forest import FeatureTypeEnum


@njit
def pre_order_traverse(
    nodes: np.ndarray,
):
    stack = []
    node_idxs = []
    current_idx = 0

    while True:
        node_idxs.append(current_idx)

        if not nodes[current_idx]["is_leaf"]:
            stack.append(nodes[current_idx]["left"])
            stack.append(nodes[current_idx]["right"])

        if not stack:
            return node_idxs
        current_idx = stack.pop(0)


@njit
def terminal_nodes(nodes: np.ndarray) -> np.ndarray:
    """Find all leaves"""

    terminal_idxs = np.argwhere(nodes["active"] & nodes["is_leaf"])
    return terminal_idxs.flatten()


@njit
def singly_internal_nodes(nodes: np.ndarray) -> np.ndarray:
    """Find all decision nodes where both children are leaves"""

    singly_internal_cond = (
        (1 - nodes["is_leaf"])
        & nodes[nodes["left"]]["is_leaf"]
        & nodes[nodes["right"]]["is_leaf"]
    )
    singly_internal_idxs = np.argwhere(nodes["active"] & singly_internal_cond)
    return singly_internal_idxs.flatten()


@njit
def get_node_subspace(
    tree: np.ndarray, node_idx: int, bounds: np.ndarray, feat_types: np.ndarray
):
    """Get the subset of the domain that reaches a given node."""
    subspace = bounds.copy()
    parent_idx = tree[node_idx]["parent"]
    while node_idx != 0:
        parent_node = tree[parent_idx]
        feature_idx = parent_node["feature_idx"]

        if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
            if node_idx == parent_node["left"]:
                subspace[feature_idx, 1] = int(parent_node["threshold"]) & int(
                    subspace[feature_idx, 1]
                )
            else:
                max_threshold = (1 << int(subspace[feature_idx, 1]).bit_length()) - 1
                neg_threshold = max_threshold - parent_node["threshold"]
                subspace[feature_idx, 1] = int(neg_threshold) & int(
                    subspace[feature_idx, 1]
                )
        else:
            if node_idx == parent_node["left"]:
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

        node_idx, parent_idx = parent_idx, tree[parent_idx]["parent"]

    return subspace
