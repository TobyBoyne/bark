import pytest
import gpytorch as gpy

from alfalfa.tree_models.forest import AlfalfaTree, AlfalfaForest, Node
from alfalfa.tree_models.tree_kernels import AFGP

def test_saving_and_loading_alternating_tree():
    tree = AlfalfaTree(depth=2)
    state = tree.state_dict()

    model = AlfalfaTree(depth=2)
    assert not model.structure_eq(tree)
    model.load_state_dict(state)
    assert model.structure_eq(tree)

def test_saving_and_loading_alternating_forest():
    forest = AlfalfaForest(depth=2, num_trees=5)
    state = forest.state_dict()

    model = AlfalfaForest(depth=2, num_trees=5)
    assert not model.structure_eq(forest)
    model.load_state_dict(state)
    assert model.structure_eq(forest)

def test_saving_and_loading_afgp():
    forest = AlfalfaForest(depth=2, num_trees=5)
    gp = AFGP(None, None, gpy.likelihoods.GaussianLikelihood(), forest)
    state = gp.state_dict()

    forest = AlfalfaForest(depth=2, num_trees=5)
    gp = AFGP(None, None, gpy.likelihoods.GaussianLikelihood(), forest)
    gp.load_state_dict(state)


def test_partial_tree():
    root = Node(
        left=Node(), right=Node(left=Node(), right=Node())
    )
    tree = AlfalfaTree(root=root)
    print(tree)
    state = tree.state_dict()

    print(state)
    model = AlfalfaTree(depth=state["_extra_state"]["depth"])
    model.load_state_dict(state)