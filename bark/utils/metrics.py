import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from jaxtyping import Float
from linear_operator.operators import DiagLinearOperator


def nlpd(
    pred_dist: torch.distributions.Distribution,
    test_y: Float[torch.Tensor, "N"],
    diag=False,
):
    """Compute the negative log predictive density of a distribution.

    Args:
        pred_dist (MVN): Predictive distribution
        diag (bool): Whether to omit (off-diagonal) correlation terms"""

    if diag:
        pred_dist = MultivariateNormal(
            pred_dist.mean, DiagLinearOperator(pred_dist.covariance_matrix.diag())
        )

    return gpytorch.metrics.negative_log_predictive_density(pred_dist, test_y)


def mse(pred_dist: torch.distributions.Distribution, test_y: torch.Tensor):
    """Compute the mean squared error of the prediction"""

    return gpytorch.metrics.mean_squared_error(pred_dist, test_y)
