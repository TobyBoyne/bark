from alfalfa.tree_models import DecisionNode, AlfalfaTree, AlfalfaForest

def test_saving_node():
    node = DecisionNode(var_idx=2, threshold=0.5)
    d = node.as_dict()
    assert d == {"var_idx": 2, "threshold": 0.5, "left": {}, "right": {}}
    new_node = DecisionNode.from_dict(d)
    assert node.structure_eq(new_node)

def test_saving_forest():
    trees = [
        AlfalfaTree(root=DecisionNode(
            var_idx=2, threshold=0.5
        )),
        AlfalfaTree(root=DecisionNode(
            var_idx=4, threshold=0.25, left=DecisionNode(var_idx=1, threshold=1.0)
        ))
    ]

    forest = AlfalfaForest(trees=trees)
    d = forest.as_dict()
    assert d == {"trees": [
        {"root": {"var_idx": 2, "threshold": 0.5, "left": {}, "right": {}}},
        {"root": {
            "var_idx": 4, "threshold": 0.25,
            "left": {"var_idx": 1, "threshold": 1.0, "left": {}, "right": {}},
            "right": {}
        }}
    ]}

    new_forest = AlfalfaForest.from_dict(d)
    assert forest.structure_eq(new_forest)