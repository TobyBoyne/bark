import jax
import jax.numpy as jnp
import pytest

from bark import forest, types
from bark.fitting.bark_prior_sampler import sample_forest
from bark.fitting.tree_proposals import grow
from bark.likelihood.likelihood import BARKLikelihood
from bark.likelihood.marginal_log_likelihood import (
    CachedGramBARKLikelihood,
    mll_bark_model,
)
from bark.testing.data_test_cases import get_continuous_data_trid

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(["likelihood_cls"], [[CachedGramBARKLikelihood]])
def test_bark_likelihoods(likelihood_cls: type[BARKLikelihood]):
    # we currently don't test the WoodburyBARKLikelihood as it is not sufficiently
    # accurate
    data = get_continuous_data_trid(N=20, dim=10)
    params = types.BARKConfig(beta=1.0)
    key = jax.random.key(0)
    random_trees = sample_forest(
        m=49, bounds=data.bounds, feat_types=data.feat_types, params=params, key=key
    )
    empty_tree = forest.create_empty_forest(m=1)
    trees = jax.tree_util.tree_map(
        lambda et, rt: jnp.concat((et, rt), axis=0), empty_tree, random_trees
    )
    bark_model = types.BARKModel(trees=trees, noise=jnp.array(0.1))
    mll = mll_bark_model(bark_model, data)

    new_tree = grow(empty_tree[0], jnp.array(0), jnp.array(0), jnp.array(0.5))
    new_trees = jax.tree_util.tree_map(
        lambda nt, rt: jnp.concat((nt[None, :], rt), axis=0), new_tree, random_trees
    )
    new_bark_model = types.BARKModel(trees=new_trees, noise=jnp.array(0.1))

    likelihood = likelihood_cls.create_from_bark_model(bark_model, data)
    likelihood_mll = likelihood.mll
    new_mll = mll_bark_model(new_bark_model, data)

    likelihood = likelihood.compute_cache_from_new_tree_proposals(
        trees, new_trees, data
    ).compute_new_tree_likelihood(bark_model, jnp.array(0), data)

    # test that the computation of the original likelihood is correct
    assert jnp.allclose(mll, likelihood_mll)

    # test that adding the tree changes the marginal log likelihood
    assert not jnp.allclose(mll, new_mll)

    # test that the cached approach gives the correct answer
    assert jnp.allclose(new_mll, likelihood.mll, rtol=1e-4)
