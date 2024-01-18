from alfalfa.tree_models.forest import AlfalfaTree, Node
from alfalfa.fitting.bart.transitions import GrowTransition

def test_grow_transition():
    tree = AlfalfaTree(root=Node())
    transition = GrowTransition(tree, tree.root, "left")
    with transition:
        assert tree.structure_eq(AlfalfaTree(root=Node(left=Node())))

    # without applying, the model should return to initial state
    assert tree.structure_eq(AlfalfaTree(root=Node()))

    with transition:
        transition.apply()

    assert tree.structure_eq(AlfalfaTree(root=Node(left=Node())))


    