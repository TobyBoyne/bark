import numpy as np
from numba import njit

S_PROP = np.array([[0.1, 0.0], [0.0, 0.5]])
S_PROP_CHOL = np.linalg.cholesky(S_PROP)
S_PROP_INV = np.linalg.inv(S_PROP)
_, S_PROP_LOGDET = np.linalg.slogdet(S_PROP)


@njit
def half_normal_logpdf(x, scale):
    # scale is std  ** 2
    log_normal = -0.5 * (x**2) / (scale) - 0.5 * np.log(scale)
    return np.where(x >= 0, log_normal, -np.inf)


@njit
def propose_positive_transition(cur_value: np.ndarray) -> np.ndarray:
    """Propose a new value for a hyperparameter that is positive.

    Proposals are made in the unconstrained log-space. Numba does not currently support multivariate normals (https://github.com/numba/numba/issues/1335)

    Args:
        cur_value (float): current value of hyperparameter
        step_size (float): size of proposed step in log-space

    Returns:
        float: proposed value
    """
    cur_log_value = np.log(cur_value + 1e-30)
    u = np.empty((2, 1), dtype=np.float64)
    for i in range(2):
        u[i] = np.random.normal()
    new_log_value = cur_log_value + S_PROP_CHOL @ u
    new_value = np.exp(new_log_value[:, 0])
    return new_value


@njit
def get_noise_scale_proposal(
    noise: float, scale: float
) -> tuple[tuple[float, float], float]:
    # TODO: consider a better sampler

    hyperparams = np.array([noise, scale])
    new_hyperparams = propose_positive_transition(hyperparams)
    new_noise, new_scale = new_hyperparams

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
