import jax
import jax.numpy as jnp

from bark import types
from bark.fitting.tree_proposals import get_tree_proposal
from bark.testing.trees_test_cases import get_continuous_trees_test_case

jax.config.update("jax_enable_x64", True)


def select_tree(trees: types.Trees, tree_index: int) -> types.Tree:
    return jax.tree_util.tree_map(lambda x: x[tree_index], trees)


def test_get_tree_proposal():
    trees_test_case = get_continuous_trees_test_case()
    bounds, feat_types = trees_test_case.bounds, trees_test_case.feat_types
    key = jax.random.key(0)

    trees = trees_test_case.trees
    tree = select_tree(trees_test_case.trees, 0)

    new_tree, tree_q_prior_ratio = get_tree_proposal(
        tree, bounds=bounds, feat_types=feat_types, params=types.BARKConfig(), key=key
    )

    assert new_tree.feature_idx.shape == tree.feature_idx.shape
    assert tree_q_prior_ratio.shape == ()
    assert tree_q_prior_ratio != -jnp.inf

    keys = jax.random.split(key, trees.feature_idx.shape[0])

    new_trees, tree_q_prior_ratios = jax.vmap(
        get_tree_proposal, in_axes=(0, None, None, None, 0)
    )(
        trees,
        bounds,
        feat_types,
        types.BARKConfig(),
        keys,
    )

    assert new_trees.feature_idx.shape == trees.feature_idx.shape
    assert tree_q_prior_ratios.shape == (trees.feature_idx.shape[0],)
    assert (tree_q_prior_ratios != -jnp.inf).all()
