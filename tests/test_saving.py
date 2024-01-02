import pytest
import gpytorch as gpy

from alfalfa.tree_models.forest import AlfalfaTree, AlfalfaForest, Node, Leaf, prune_tree_hook
from alfalfa.tree_models.tree_kernels import AFGP

def test_saving_and_loading_alternating_tree():
    tree = AlfalfaTree(depth=2)
    tree.initialise_tree([0])
    state = tree.state_dict()

    model = AlfalfaTree(depth=2)
    model.initialise_tree([0])
    assert not model.structure_eq(tree)
    model.load_state_dict(state)
    assert model.structure_eq(tree)

def test_saving_and_loading_alternating_forest():
    forest = AlfalfaForest(depth=2, num_trees=5)
    forest.initialise_forest([0])
    state = forest.state_dict()

    model = AlfalfaForest(depth=2, num_trees=5)
    model.initialise_forest([0])
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
        left=Leaf(), right=Node(left=Leaf(), right=Leaf())
    )
    tree = AlfalfaTree(root=root)
    state = tree.state_dict()

    model = AlfalfaTree(depth=state["_extra_state"]["depth"])
    model.register_load_state_dict_post_hook(prune_tree_hook)
    model.load_state_dict(state)