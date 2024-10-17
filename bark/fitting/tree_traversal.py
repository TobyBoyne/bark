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
    tree: np.ndarray, node_idx: int, bounds: list[list[float]], feat_types: np.ndarray
):
    """Get the subset of the domain that reaches a given node."""
    subspace = [[b for b in bounds[i]] for i in range(len(bounds))]
    parent_idx = tree[node_idx]["parent"]
    while parent_idx != -1:
        node = tree[node_idx]
        parent_node = tree[parent_idx]
        feature_idx = parent_node["feature_idx"]

        if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
            pass
            # if node_idx == parent_node["left"]:
            #     subspace[feature_idx] = [node["threshold"]]
            # else:
            #     subspace[feature_idx] = [b for b in subspace[feature_idx] if b != node["threshold"]]
        else:
            if node_idx == parent_node["left"]:
                subspace[feature_idx][1] = node["threshold"]
            else:
                subspace[feature_idx][0] = node["threshold"]

    return subspace
