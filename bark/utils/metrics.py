import numpy as np
from jaxtyping import Float


def gaussian_log_likelihood(
    mu: Float[np.ndarray, "N batch"],
    var: Float[np.ndarray, "N batch"],
    y: Float[np.ndarray, "N batch"],
) -> Float[np.ndarray, "N batch"]:
    """Compute the log likelihood of a Gaussian distribution.

    Args:
        mu (np.ndarray): The mean of the distribution
        var (np.ndarray): The variance of the distribution
        y (np.ndarray): The observed values"""

    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var


def nlpd(
    mu: Float[np.ndarray, "N batch"],
    var: Float[np.ndarray, "N batch"],
    test_y: Float[np.ndarray, "N 1"],
    diag=False,
):
    """Compute the negative log predictive density of a Normal distribution.

    Args:
        mu (np.ndarray): The mean of the distribution
        var (np.ndarray): The variance of the distribution
        diag (bool): Whether to omit (off-diagonal) correlation terms"""

    return -gaussian_log_likelihood(mu, var, test_y).sum(axis=0) / test_y.shape[0]


def mse(mu: Float[np.ndarray, "N batch"], test_y: Float[np.ndarray, "N 1"]):
    """Compute the mean squared error of the prediction"""

    return np.mean(np.square(mu - test_y))
