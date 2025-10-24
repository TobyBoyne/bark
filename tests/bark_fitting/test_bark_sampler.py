import jax

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
    samples = jax.jit(run_bark_sampler, static_argnames=("params", "seed"))(
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


test_bark_sampler()
