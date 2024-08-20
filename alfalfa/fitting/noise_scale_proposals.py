import numpy as np
from numba import njit

S_PROP = np.array([[0.1, 0.0], [0.0, 0.5]])
S_PROP_INV = np.linalg.inv(S_PROP)
_, S_PROP_LOGDET = np.linalg.slogdet(S_PROP)


@njit
def get_noise_proposal(forest: np.ndarray, noise, scale, rng: np.random.Generator):
    pass


@njit
def get_scale_proposal(forest: np.ndarray, noise, scale, rng: np.random.Generator):
    pass


@njit
def half_normal_logpdf(x, scale):
    # scale is std  ** 2
    log_normal = -0.5 * (x**2) / (scale) - 0.5 * np.log(scale)
    return np.where(x >= 0, log_normal, -np.inf)


@njit
def propose_positive_transition(
    cur_value: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Propose a new value for a hyperparameter that is positive.

    Proposals are made in the unconstrained log-space

    Args:
        cur_value (float): current value of hyperparameter
        step_size (float): size of proposed step in log-space

    Returns:
        float: proposed value
    """
    cur_log_value = np.log(cur_value + 1e-30)
    new_log_value = rng.multivariate_normal(mean=cur_log_value, cov=S_PROP)
    new_value = np.exp(new_log_value)
    return new_value


@njit
def get_noise_scale_proposal(
    forest: np.ndarray, noise, scale, rng: np.random.Generator
) -> tuple[tuple[float, float], float]:
    # TODO: consider a better sampler

    hyperparams = np.array([noise, scale])
    new_hyperparams = propose_positive_transition(hyperparams, rng)
    new_noise, new_scale = new_hyperparams

    # random walk in the log-space is symmetric
    log_q = 0.0
    log_prior = (
        half_normal_logpdf(new_noise, 1.0)
        + half_normal_logpdf(new_scale, 5.0)
        - half_normal_logpdf(noise, 1.0)
        - half_normal_logpdf(scale, 5.0)
    )

    log_q_prior = log_q + log_prior
    return (new_noise, new_scale), log_q_prior
