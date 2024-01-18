from alfalfa.tree_models.forest import AlfalfaTree, Node
import alfalfa.fitting.bart.tree_traversal as traverse

def test_terminal_nodes():
    n1 = Node()
    n2 = Node(left=n1)
    tree = AlfalfaTree(root=n2)
    g = traverse.terminal_nodes(tree)
    assert g == [n1.left, n1.right, n2.right]

def test_singly_internal_nodes():
    n1 = Node()
    n2 = Node(left=n1)
    tree = AlfalfaTree(root=n2)
    g = traverse.singly_internal_nodes(tree)
    assert g == [n1]