import jax
import jax.numpy as jnp

from bark import forest, types
from bark.fitting.bark_prior_sampler import sample_forest
from bark.fitting.marginal_log_likelihood import (
    get_cached_grams,
    mll_bark_model,
    mll_bark_model_cached_gram,
)
from bark.fitting.tree_proposals import grow
from bark.testing.data_test_cases import get_continuous_data_trid

jax.config.update("jax_enable_x64", True)


def test_mll_bark_model_cached_gram():
    data = get_continuous_data_trid(N=100, dim=10)
    params = types.BARKConfig()
    key = jax.random.key(0)
    random_trees = sample_forest(
        m=49, bounds=data.bounds, feat_types=data.feat_types, params=params, key=key
    )
    empty_tree = forest.create_empty_forest(m=1)
    trees = jax.tree_util.tree_map(
        lambda et, rt: jnp.concat((et, rt), axis=0), empty_tree, random_trees
    )
    bark_model = types.BARKModel(trees=trees, noise=jnp.array(0.1))

    new_tree = grow(empty_tree[0], jnp.array(0), jnp.array(0), jnp.array(0.5))
    new_trees = jax.tree_util.tree_map(
        lambda nt, rt: jnp.concat((nt[None, :], rt), axis=0), new_tree, random_trees
    )
    new_bark_model = types.BARKModel(trees=new_trees, noise=jnp.array(0.1))

    mll = mll_bark_model(bark_model, data)
    new_mll = mll_bark_model(new_bark_model, data)

    G_XX, G_XX_delta = get_cached_grams(trees, new_trees, data)
    new_mll_cached_grams = mll_bark_model_cached_gram(
        new_bark_model, G_XX, G_XX_delta[..., 0], data
    )

    # test that adding the tree changes the marginal log likelihood
    assert not jnp.allclose(mll, new_mll)

    # test that the cached approach gives the correct answer
    assert jnp.allclose(new_mll, new_mll_cached_grams, rtol=1e-4)
