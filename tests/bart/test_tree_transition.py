from alfalfa.tree_models.forest import AlfalfaTree, DecisionNode
from alfalfa.fitting.bart.transitions import GrowTransition

def test_grow_transition():
    tree = AlfalfaTree(root=DecisionNode())
    transition = GrowTransition(tree, tree.root, "left")
    with transition:
        assert tree.structure_eq(AlfalfaTree(root=DecisionNode(left=DecisionNode())))

    # without applying, the model should return to initial state
    assert tree.structure_eq(AlfalfaTree(root=DecisionNode()))

    with transition:
        transition.apply()

    assert tree.structure_eq(AlfalfaTree(root=DecisionNode(left=DecisionNode())))


    