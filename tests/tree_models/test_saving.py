import gpytorch
import pytest

from bark.forest import BARKForest, BARKTree, DecisionNode
from bark.tree_kernels import BARKGP


@pytest.fixture
def node1():
    return (
        DecisionNode(var_idx=2, threshold=0.5),
        {"var_idx": 2, "threshold": 0.5, "left": {}, "right": {}},
    )


@pytest.fixture
def node2():
    return (
        DecisionNode(
            var_idx=4, threshold=0.25, left=DecisionNode(var_idx=1, threshold=1.0)
        ),
        {
            "var_idx": 4,
            "threshold": 0.25,
            "left": {"var_idx": 1, "threshold": 1.0, "left": {}, "right": {}},
            "right": {},
        },
    )


def test_saving_node(node1):
    node, expected_d = node1
    d = node.as_dict()
    assert d == expected_d
    new_node = DecisionNode.from_dict(d)
    assert node.structure_eq(new_node)


def test_saving_forest(node1, node2):
    root1, expected_d1 = node1
    root2, expected_d2 = node2
    trees = [BARKTree(root=root1), BARKTree(root=root2)]

    forest = BARKForest(trees=trees)
    d = forest.as_dict()
    assert d == {
        "tree_model_type": "forest",
        "trees": [
            {"tree_model_type": "tree", "root": expected_d1},
            {"tree_model_type": "tree", "root": expected_d2},
        ],
    }

    new_forest = BARKForest.from_dict(d)
    assert forest.structure_eq(new_forest)


def test_saving_tree_gp(node1, node2):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    root1, expected_d1 = node1
    root2, expected_d2 = node2
    trees = [BARKTree(root=root1), BARKTree(root=root2)]

    forest = BARKForest(trees=trees)
    gp = BARKGP(None, None, likelihood, forest)

    state = gp.state_dict()

    gp2 = BARKGP(None, None, likelihood, None)
    gp2.load_state_dict(state)
    assert state == gp2.state_dict()
