from alfalfa.tree_models.forest import AlfalfaTree, DecisionNode, LeafNode

def test_tree_structure_equals():
    tree1 = DecisionNode(var_idx=0, threshold=0.5)

    tree2 = DecisionNode(var_idx=0, threshold=0.5)

    tree3 = DecisionNode(
        var_idx=0, threshold=0.5,
        right=DecisionNode(var_idx=1, threshold=1.0)
    )

    assert tree1.structure_eq(tree2)
    assert not tree1.structure_eq(tree3)