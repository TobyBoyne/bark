import jax

from bark import types
from bark.fitting.bark_sampler import run_bark_sampler
from bark.testing.bark_test_cases import get_continuous_bark_test_case

jax.config.update("jax_enable_x64", True)


def test_bark_sampler():
    bark_test_case = get_continuous_bark_test_case()

    samples = run_bark_sampler(
        bark_test_case.bark_model,
        data=bark_test_case.data,
        params=types.BARKConfig(),
        seed=0,
    )
    print(samples)


test_bark_sampler()
