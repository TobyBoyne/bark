import numpy as np
from numba import njit

from alfalfa.forest_numba import NODE_RECORD_DTYPE


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


nodes = np.array(
    [
        (0, 0, 0.5, 1, 2, 0, 1),
        (0, 0, 0.25, 3, 4, 1, 1),
        (1, 0, 1.0, 0, 0, 1, 1),
        (1, 0, 1.0, 0, 0, 2, 1),
        (1, 0, 1.0, 0, 0, 2, 1),
    ],
    dtype=NODE_RECORD_DTYPE,
)

print(terminal_nodes(nodes))
print(singly_internal_nodes(nodes))
