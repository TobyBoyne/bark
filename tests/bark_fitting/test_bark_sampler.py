import jax
import jax.numpy as jnp

from bark import types
from bark.fitting.bark_sampler import run_bark_sampler
from bark.testing.bark_test_cases import get_continuous_bark_test_case

jax.config.update("jax_enable_x64", True)


def test_bark_sampler():
    bark_test_case = get_continuous_bark_test_case()
    params = types.BARKConfig(
        num_samples=4,
        warmup_steps=30,
        steps_per_sample=12,
    )
    samples = run_bark_sampler(
        bark_test_case.bark_model,
        data=bark_test_case.data,
        params=params,
        seed=0,
    )

    assert isinstance(samples, types.BARKModel)
    assert samples.num_trees == bark_test_case.bark_model.num_trees
    assert samples.noise.shape == (params.num_samples,)
    assert samples.trees.feature_idx.shape == (
        params.num_samples,
        *bark_test_case.bark_model.trees.feature_idx.shape,
    )

    bark_model_parallel = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, reps=(params.num_chains, *[1 for _ in x.shape])),
        bark_test_case.bark_model,
    )

    samples_parallel = jax.vmap(run_bark_sampler, in_axes=(0, None, None, None))(
        bark_model_parallel,
        bark_test_case.data,
        params,
        0,
    )
    assert isinstance(samples_parallel, types.BARKModel)
    assert samples_parallel.num_trees == bark_test_case.bark_model.num_trees
    assert samples_parallel.noise.shape == (
        params.num_chains,
        params.num_samples,
    )
    assert samples_parallel.trees.feature_idx.shape == (
        params.num_chains,
        params.num_samples,
        *bark_test_case.bark_model.trees.feature_idx.shape,
    )

    # check that the random sampling is random, and that each chain isn't identical
    assert jnp.allclose(
        samples_parallel.trees.threshold[0], samples_parallel.trees.threshold[1]
    )
