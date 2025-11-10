from dataclasses import replace

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
        num_chains=1,
    )
    samples = run_bark_sampler(
        bark_test_case.bark_model,
        data=bark_test_case.data,
        params=params,
        key=jax.random.key(0),
    )

    assert isinstance(samples, types.BARKModel)
    assert samples.num_trees == bark_test_case.bark_model.num_trees
    assert samples.noise_var.shape == (params.num_samples,)
    assert samples.trees.feature_idx.shape == (
        params.num_samples,
        *bark_test_case.bark_model.trees.feature_idx.shape,
    )

    params = replace(params, num_chains=2)

    samples_parallel = run_bark_sampler(
        bark_test_case.bark_model,
        data=bark_test_case.data,
        params=params,
        key=jax.random.key(0),
    )
    assert isinstance(samples_parallel, types.BARKModel)
    assert samples_parallel.num_trees == bark_test_case.bark_model.num_trees
    assert samples_parallel.noise_var.shape == (
        params.num_chains,
        params.num_samples,
    )
    assert samples_parallel.trees.feature_idx.shape == (
        params.num_chains,
        params.num_samples,
        *bark_test_case.bark_model.trees.feature_idx.shape,
    )

    # check that the random sampling is random, and that each chain isn't identical
    assert not jnp.allclose(
        samples_parallel.trees.threshold[0], samples_parallel.trees.threshold[1]
    )
    assert not jnp.allclose(
        samples_parallel.noise_var[0], samples_parallel.noise_var[1]
    )
