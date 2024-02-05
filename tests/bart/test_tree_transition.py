from alfalfa.fitting.bart.tree_transitions import GrowTransition
from alfalfa.forest import AlfalfaTree, DecisionNode


def test_grow_transition():
    tree = AlfalfaTree(root=DecisionNode())
    transition = GrowTransition(tree, tree.root.left, DecisionNode())
    with transition:
        assert tree.structure_eq(AlfalfaTree(root=DecisionNode(left=DecisionNode())))

    # context manager correctly applies inverse transition upon exit
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode()))

    transition.apply()
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode(left=DecisionNode())))
