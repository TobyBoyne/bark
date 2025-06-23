from typing import TYPE_CHECKING

import numpy as np
import scipy.special as special
from numba import njit

if TYPE_CHECKING:
    from bark.fitting.bark_sampler import BARKTrainParamsNumba

PROPOSAL_STEP_SIZE = np.array([0.000001, 0.000002])
PROPOSAL_STEP_SIZE = np.array([1.0, 0.00000001])


@njit
def half_normal_logpdf(x, scale):
    # scale is std  ** 2
    log_normal = -0.5 * (x**2) / (scale) - 0.5 * np.log(scale)
    return np.where(x >= 0, log_normal, -np.inf)


@njit
def gamma_logpdf(x, shape, rate):
    return (
        (shape - 1) * np.log(x)
        - rate * x
        - special.gammaln(shape)
        + shape * np.log(rate)
    )


@njit
def inverse_gamma_logpdf(x, shape, rate):
    scale = 1 / rate
    return (
        -(shape + 1) * np.log(x)
        - scale / x
        - special.gammaln(shape)
        + shape * np.log(scale)
    )


@njit
def propose_positive_transition(cur_value: float, step_size: float) -> float:
    """Propose a new value for a hyperparameter that is positive.

    Proposals are made in the unconstrained log-space. Numba does not currently support multivariate normals (https://github.com/numba/numba/issues/1335)

    Args:
        cur_value (float): current value of hyperparameter

    Returns:
        float: proposed value
    """
    cur_log_value = np.log(cur_value + 1e-30)
    u = np.random.normal()
    new_log_value = cur_log_value + step_size * u
    new_value = np.exp(new_log_value)
    return new_value


@njit
def propose_positive_transition_softplus(cur_value: float, step_size: float) -> float:
    cur_transformed_value = np.log(np.exp(cur_value) - 1)
    u = np.random.normal()
    new_transformed_value = cur_transformed_value + step_size * u
    new_value = np.log(np.exp(new_transformed_value) + 1)
    return new_value


@njit
def get_noise_scale_proposal(
    noise: float, scale: float, params: "BARKTrainParamsNumba"
) -> tuple[tuple[float, float], float]:
    # TODO: consider a better sampler
    if params.use_softplus_transform:
        return get_noise_scale_proposal_softplus(noise, scale, params)

    if not params.sample_scale:
        raise NotImplementedError(
            "You must sample the scale parameter in the log space"
        )

    new_noise = propose_positive_transition(noise, PROPOSAL_STEP_SIZE[0])
    new_scale = propose_positive_transition(scale, PROPOSAL_STEP_SIZE[1])

    # random walk in the log-space mostly cancels
    # https://stats.stackexchange.com/a/297559
    log_q = -np.log(noise) - np.log(scale) + np.log(new_noise) + np.log(new_scale)
    log_prior = (
        half_normal_logpdf(new_noise, 1.0)
        + half_normal_logpdf(new_scale, 5.0)
        - half_normal_logpdf(noise, 1.0)
        - half_normal_logpdf(scale, 5.0)
    )

    log_q_prior = log_q + log_prior
    return (new_noise, new_scale), log_q_prior


@njit
def get_noise_scale_proposal_softplus(
    noise: float, scale: float, params: "BARKTrainParamsNumba"
) -> tuple[tuple[float, float], float]:
    if not params.sample_scale:
        new_noise, log_q_prior = get_noise_proposal_softplus(noise, params)
        return (new_noise, scale), log_q_prior

    new_noise = propose_positive_transition_softplus(noise, PROPOSAL_STEP_SIZE[0])
    new_scale = propose_positive_transition_softplus(scale, PROPOSAL_STEP_SIZE[1])

    noise_step_var, scale_step_var = PROPOSAL_STEP_SIZE**2
    log_q = (
        (np.log(np.exp(noise) - 1) - np.log(np.exp(new_noise) - 1)) ** 2
        / noise_step_var
        + np.log(1 - np.exp(-noise))
        - np.log(1 - np.exp(-new_noise))
        + (np.log(np.exp(scale) - 1) - np.log(np.exp(new_scale) - 1)) ** 2
        / scale_step_var
        + np.log(1 - np.exp(-scale))
        - np.log(1 - np.exp(-new_scale))
    )

    log_prior = (
        half_normal_logpdf(new_noise, 1.0)
        + half_normal_logpdf(new_scale, 5.0)
        - half_normal_logpdf(noise, 1.0)
        - half_normal_logpdf(scale, 5.0)
    )

    log_q_prior = log_q + log_prior
    return (new_noise, new_scale), log_q_prior


@njit
def get_noise_proposal_softplus(
    noise: float, params: "BARKTrainParamsNumba"
) -> tuple[float, float]:
    noise_step = PROPOSAL_STEP_SIZE[0]
    new_noise = propose_positive_transition_softplus(noise, noise_step)

    noise_step_var = noise_step**2
    log_q = (
        (np.log(np.exp(noise) - 1) - np.log(np.exp(new_noise) - 1)) ** 2
        / noise_step_var
        + np.log(1 - np.exp(-noise))
        - np.log(1 - np.exp(-new_noise))
    )

    log_q *= -1

    log_prior = inverse_gamma_logpdf(
        new_noise, params.gamma_prior_shape, params.gamma_prior_rate
    ) - inverse_gamma_logpdf(noise, params.gamma_prior_shape, params.gamma_prior_rate)

    log_q_prior = log_q + log_prior
    return new_noise, log_q_prior
