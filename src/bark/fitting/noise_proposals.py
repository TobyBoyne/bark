import jax
import jax.numpy as jnp
import jax.scipy.special as special
from jaxtyping import Array, Float

from bark import types


def inverse_gamma_logpdf(x: Float[Array, "..."], shape: float, rate: float):
    scale = 1 / rate
    return (
        -(shape + 1) * jnp.log(x)
        - scale / x
        - special.gammaln(shape)
        + shape * jnp.log(scale)
    )


def propose_positive_transition_softplus(
    cur_value: Float[Array, ""], step_size: float, key: jax.Array
) -> Float[Array, ""]:
    cur_transformed_value = jnp.log(jnp.exp(cur_value) - 1)
    u = jax.random.normal(key, cur_value.shape, dtype=cur_value.dtype)
    new_transformed_value = cur_transformed_value + step_size * u
    new_value = jnp.logaddexp(new_transformed_value, 0)
    return new_value


def compute_log_q_noise(
    noise: Float[Array, ""], new_noise: Float[Array, ""]
) -> Float[Array, ""]:
    return jnp.log(1 - jnp.exp(-new_noise)) - jnp.log(1 - jnp.exp(-noise))


def compute_log_prior_noise(
    noise: Float[Array, ""], new_noise: Float[Array, ""], params: types.BARKConfig
) -> Float[Array, ""]:
    return inverse_gamma_logpdf(
        new_noise, params.gamma_prior_shape, params.gamma_prior_rate
    ) - inverse_gamma_logpdf(noise, params.gamma_prior_shape, params.gamma_prior_rate)


def get_noise_proposal_softplus(
    noise: Float[Array, ""], params: types.BARKConfig, key: jax.Array
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    noise_step = params.noise_step_size
    new_noise = propose_positive_transition_softplus(noise, noise_step, key)

    log_q = compute_log_q_noise(noise, new_noise)
    log_prior = compute_log_prior_noise(noise, new_noise, params)

    log_q_prior = log_q + log_prior
    return new_noise, log_q_prior
