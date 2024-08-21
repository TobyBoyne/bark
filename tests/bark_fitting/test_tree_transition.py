from alfalfa.fitting.bart.tree_transitions import (
    ChangeTransition,
    GrowTransition,
    PruneTransition,
)
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


def test_prune_transition():
    tree = AlfalfaTree(root=DecisionNode(left=DecisionNode()))
    transition = PruneTransition(tree, tree.root.left)
    with transition:
        assert tree.structure_eq(AlfalfaTree(root=DecisionNode()))

    # context manager correctly applies inverse transition upon exit
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode(left=DecisionNode())))

    transition.apply()
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode()))


def test_change_transition():
    tree = AlfalfaTree(root=DecisionNode(var_idx=0, threshold=0.5))
    transition = ChangeTransition(tree, tree.root, 1, 2.0)
    with transition:
        assert tree.structure_eq(
            AlfalfaTree(root=DecisionNode(var_idx=1, threshold=2.0))
        )

    # context manager correctly applies inverse transition upon exit
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode(var_idx=0, threshold=0.5)))

    transition.apply()
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode(var_idx=1, threshold=2.0)))
