import jax
import jax.numpy as jnp
import scipy.special as special
from jaxtyping import Array, Float

from bark import types


def inverse_gamma_logpdf(x, shape, rate):
    scale = 1 / rate
    return (
        -(shape + 1) * jnp.log(x)
        - scale / x
        - special.gammaln(shape)
        + shape * jnp.log(scale)
    )


def propose_positive_transition_softplus(
    cur_value: Float[Array, "..."], step_size: Float[Array, ""], key: jax.Array
) -> Float[Array, "..."]:
    cur_transformed_value = jnp.log(jnp.exp(cur_value) - 1)
    u = jax.random.normal(key, cur_value.shape, dtype=cur_value.dtype)
    new_transformed_value = cur_transformed_value + step_size * u
    new_value = jnp.logaddexp(new_transformed_value, 0)
    return new_value


def get_noise_proposal_softplus(
    noise: Float[Array, "..."], params: types.BARKConfig, key: jax.Array
) -> tuple[Float[Array, "..."], Float[Array, ""]]:
    noise_step = jnp.array(1.0)
    new_noise = propose_positive_transition_softplus(noise, noise_step, key)

    noise_step_var = noise_step**2
    log_q = -(
        (jnp.log(jnp.exp(noise) - 1) - jnp.log(jnp.exp(new_noise) - 1)) ** 2
        / noise_step_var
        + jnp.log(1 - jnp.exp(-noise))
        - jnp.log(1 - jnp.exp(-new_noise))
    )

    log_prior = inverse_gamma_logpdf(
        new_noise, params.gamma_prior_shape, params.gamma_prior_rate
    ) - inverse_gamma_logpdf(noise, params.gamma_prior_shape, params.gamma_prior_rate)

    log_q_prior = log_q + log_prior
    return new_noise, log_q_prior
