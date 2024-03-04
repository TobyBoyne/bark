import gpytorch
import numpy as np
import torch

from ..tree_kernels import AlfalfaMOGP
from .thompson_sampling import generate_fstar_samples

standard_normal = torch.distributions.Normal(loc=0, scale=1)
SQRT_2PI_E = torch.sqrt(2 * torch.pi * torch.exp(torch.tensor(1)))


def choose_fidelity_information_based(model: AlfalfaMOGP, x: torch.Tensor) -> int:
    """Choose the fidelity level for a given input.

    Args:
        model (AlfalfaMOGP): Multi-output GP
        x (torch.Tensor): Proposed x

    Returns:
        int: fidelity that provides the greatest information gain
    """

    f_star = generate_fstar_samples(model, num_samples=100)
    igs = [
        information_gain(model, x, f_star, fidelity)
        for fidelity in range(model.num_tasks)
    ]
    fidelity = np.argmax(igs)
    return fidelity


def information_gain(
    model: AlfalfaMOGP, x: torch.Tensor, f_star: torch.Tensor, fidelity: int
):
    fidelity_vector = torch.tensor(fidelity)
    posterior = model(x, fidelity_vector)
    mu_m, sigma_m = posterior.mean, posterior.stddev
    H_1 = torch.log(sigma_m * SQRT_2PI_E)
    if fidelity == 0:
        H_2 = _entropy_target_fidelity(mu_m, sigma_m, f_star)
    else:
        H_2 = _entropy_low_fidelity(mu_m, sigma_m, f_star, model, x, fidelity_vector)

    H = H_1 - H_2

    assert not H.isnan().any()
    assert not H.isinf().any()
    return H


def _entropy_target_fidelity(
    mu_m: torch.Tensor, sigma_m: torch.Tensor, f_star: torch.Tensor
):
    # calculate expected entropy of f(X, m) | f_*, D_t
    # calculate gamma
    gamma = (f_star - mu_m) / (sigma_m + 1e-7)
    # cdf and pdf terms
    cdf_term = standard_normal.cdf(gamma)
    pdf_term = torch.exp(standard_normal.log_prob(gamma))
    # finally calculate entropy
    # make sure value inside log is non-zero for numerical stability using masked_fill
    inner_log = SQRT_2PI_E * sigma_m * cdf_term
    log_term = torch.log(inner_log.masked_fill(inner_log <= 0, 1e-10))
    # second term
    second_term = gamma * pdf_term / (2 * cdf_term + 1e-10)
    # finally take Monte Carlo Estimate
    H_2_samples = log_term - second_term
    H_2 = H_2_samples.mean(dim=1).reshape(-1, 1)
    return H_2


def _entropy_low_fidelity(
    mu_m: torch.Tensor,
    sigma_m: torch.Tensor,
    f_star: torch.Tensor,
    model: AlfalfaMOGP,
    x: torch.Tensor,
    fidelity_vector: torch.Tensor,
):
    # define fidelity vectors
    target_fidelity_vector = torch.tensor(0)
    joint_fidelity_vector = torch.concat((fidelity_vector, target_fidelity_vector))
    # obtain joint covariance matrix
    with gpytorch.settings.fast_pred_var():
        out = model(x.repeat(2, 1), joint_fidelity_vector)
        covar_matrix = out.lazy_covariance_matrix
    # obtain target fidelity mean vector
    posterior = model(x, target_fidelity_vector)
    mu_0, sigma_0 = posterior.mean, posterior.stddev
    # obtain smaller covariance matrix
    # batch_size is 1 in this setting (not doing batch proposals... yet)
    batch_size = 1
    covar_matrix_mM = covar_matrix[:batch_size, batch_size:]
    covar_matrix_MM = covar_matrix[batch_size:, batch_size:]
    # obtain variances
    sigma_mM_sqrd = covar_matrix_mM.diag().reshape(-1, 1)
    sigma_M_sqrd = covar_matrix_MM.diag().reshape(-1, 1)
    # define s^2
    s_sqrd = sigma_M_sqrd - (sigma_mM_sqrd) ** 2 / (sigma_m**2 + 1e-9)

    # now we can define Psi(x)
    def Psi(f):
        u_x = mu_0 + sigma_mM_sqrd * (f - mu_m) / (
            sigma_m**2 + 1e-9
        )  # should be size: batch size x 1
        # cdf and pdf terms
        cdf_term = standard_normal.cdf(
            (f_star - u_x) / (torch.sqrt(s_sqrd) + 1e-9)
        )  # should be size: batch size x samples
        pdf_term = torch.exp(standard_normal.log_prob((f - mu_m) / (sigma_m + 1e-9)))
        return cdf_term * pdf_term

    # and define Z, add 1e-10 for numerical stability
    inv_Z = standard_normal.cdf((f_star - mu_0) / (sigma_0 + 1e-9)) * sigma_m + 1e-10
    Z = 1 / inv_Z
    # we can now estimate the one dimensional integral
    # define integral range

    min_integration = -10.0
    max_integration = 10.0
    num_of_integration_steps = 250

    f_range = torch.linspace(
        min_integration, max_integration, steps=num_of_integration_steps
    )
    # preallocate the space
    integral_grid = torch.zeros((num_of_integration_steps, batch_size, f_star.shape[0]))
    # calculate corresponding y values
    for idx, f in enumerate(f_range):
        z_psi = Z * Psi(f)
        # recall that limit of x * log(x) as x-> 0 is 0; but computationally we get nans, so set it to 1 to obtain correct values
        z_psi = z_psi.masked_fill(z_psi <= 0, 1)
        y_vals = z_psi * torch.log(z_psi)
        assert not y_vals.isnan().any()
        integral_grid[idx, :, :] = y_vals
    # estimate integral using trapezium rule
    integral_estimates = torch.trapezoid(integral_grid, f_range, dim=0)
    # now estimate H2 using Monte Carlo
    H_2 = -integral_estimates.mean(dim=1).reshape(-1, 1)
    return H_2
