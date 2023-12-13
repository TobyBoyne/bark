from alfalfa.tree_models.forest import AlfalfaTree, Node, Leaf

def test_tree_structure_equals():
    tree1 = Node(var_idx=0, threshold=0.5)

    tree2 = Node(var_idx=0, threshold=0.5)

    tree3 = Node(
        var_idx=0, threshold=0.5,
        right=Node(var_idx=1, threshold=1.0)
    )

    assert tree1.structure_eq(tree2)
    assert not tree1.structure_eq(tree3)