import jax
import jax.numpy as jnp
import jax.scipy.special as special
import pytest

from bark import types
from bark.fitting.noise_proposals import (
    compute_log_prior_noise,
    compute_log_q_noise,
    get_noise_proposal_softplus,
    inverse_gamma_logpdf,
)


@pytest.mark.parametrize(
    ["gamma_rate", "gamma_shape"],
    [
        [1.0, 1.0],
        [1.5, 0.6],
        [0.4, 1.8],
    ],
)
def test_inverse_gamma_logpdf(gamma_rate, gamma_shape):
    key = jax.random.key(0)
    gamma_scale = 1 / gamma_rate
    gamma_samples = jax.random.gamma(key, gamma_shape, (10_000,))
    inv_gamma_samples = (1 / gamma_samples) * gamma_scale

    bin_edges = jnp.linspace(0.1, 1.4, num=20)
    # compute Z = P(lb < X < ub) normalization
    gamaincc = lambda x: special.gammaincc(gamma_shape, gamma_scale / x)
    Z = gamaincc(bin_edges[-1]) - gamaincc(bin_edges[0])

    hist = jnp.histogram(inv_gamma_samples, bins=bin_edges, density=True)
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    logpdf = inverse_gamma_logpdf(x, gamma_shape, gamma_rate)
    conditional_pdf = jnp.exp(logpdf) / Z

    # check that the root mean squared deviation from the expected pdf is small
    rmse = jnp.sqrt(jnp.square(hist[0] - conditional_pdf).mean()).item()
    assert rmse < 0.05


def test_get_noise_proposal_softplus():
    keys = jax.random.split(jax.random.key(0), num=10_000)
    initial_noise = jnp.array(1.0)
    params = types.BARKConfig(
        noise_step_size=1.0, gamma_prior_rate=5.0, gamma_prior_shape=1.5
    )

    def loop(noise, key):
        proposed_noise, log_q_prior = get_noise_proposal_softplus(noise, params, key)
        accept = jnp.log(jax.random.uniform(key)) <= log_q_prior
        new_noise = jnp.where(accept, proposed_noise, noise)
        return new_noise, {
            "noise": noise,
            "proposed_noise": proposed_noise,
            "accept": accept,
            "log_q_prior": log_q_prior,
        }

    _, ys = jax.lax.scan(loop, initial_noise, keys)
    assert 0.2 <= ys["accept"].mean().item() <= 0.8

    # test symmetry of proposal
    new_noise = jnp.array(0.5)
    log_q_prior_fwd = compute_log_q_noise(
        initial_noise, new_noise
    ) + compute_log_prior_noise(initial_noise, new_noise, params)
    log_q_prior_bwd = compute_log_q_noise(
        new_noise, initial_noise
    ) + compute_log_prior_noise(new_noise, initial_noise, params)
    assert jnp.allclose(log_q_prior_fwd, -log_q_prior_bwd)
