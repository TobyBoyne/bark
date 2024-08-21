import numpy as np

import alfalfa.fitting.bart.tree_traversal as traverse
from alfalfa.forest import AlfalfaTree, DecisionNode
from alfalfa.forest_numba import NODE_RECORD_DTYPE


def test_terminal_nodes():
    n1 = DecisionNode()
    n2 = DecisionNode(left=n1)
    tree = AlfalfaTree(root=n2)
    g = traverse.terminal_nodes(tree)
    assert g == [n1.left, n1.right, n2.right]


def test_singly_internal_nodes():
    n1 = DecisionNode()
    n2 = DecisionNode(left=n1)
    tree = AlfalfaTree(root=n2)
    g = traverse.singly_internal_nodes(tree)
    assert g == [n1]


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
