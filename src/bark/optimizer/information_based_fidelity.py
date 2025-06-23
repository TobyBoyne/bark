import warnings

import gpytorch
import numpy as np
import torch
from beartype.cave import IntType
from jaxtyping import Float, Int, Shaped

from ..tree_kernels import BARKMOGP
from .thompson_sampling import generate_fstar_samples

standard_normal = torch.distributions.Normal(loc=0, scale=1)
SQRT_2PI_E = torch.sqrt(2 * torch.pi * torch.exp(torch.tensor(1)))


def propose_fidelity_information_based(
    model: BARKMOGP, x: Shaped[torch.Tensor, "1 D"], costs: list[float]
) -> IntType:
    """Choose the fidelity level for a given input.

    Args:
        model (BARKMOGP): Multi-output GP
        x (torch.Tensor): Proposed x

    Returns:
        int: fidelity that provides the greatest information gain
    """

    f_star = generate_fstar_samples(model, num_samples=100)
    with torch.no_grad():
        igs = [
            information_gain(model, x, f_star, fidelity).item() / costs[fidelity]
            for fidelity in range(model.num_tasks)
        ]
    fidelity = np.argmax(igs)
    return fidelity


def information_gain(
    model: BARKMOGP,
    x: Shaped[torch.Tensor, "batch D"],
    f_star: Float[torch.Tensor, "N"],
    fidelity: int,
) -> Float[torch.Tensor, "1 1"]:
    fidelity_vector = torch.full((x.shape[0], 1), fidelity)
    posterior = model(x, fidelity_vector)
    mu_m, sigma_m = posterior.mean, posterior.stddev
    mu_m, sigma_m = mu_m.reshape(-1, 1), sigma_m.reshape(-1, 1)
    H_1 = torch.log(sigma_m * SQRT_2PI_E)
    if fidelity == 0:
        H_2 = _entropy_target_fidelity(mu_m, sigma_m, f_star)
    else:
        H_2 = _entropy_low_fidelity(mu_m, sigma_m, f_star, model, x, fidelity_vector)

    H = H_1 - H_2

    assert not H.isnan().any()
    assert not H.isinf().any()

    if H < 0:
        warnings.warn(
            f"Negative information gain: {H.item()}. This is likely due to numerical instability/inaccuracy in integration."
        )
    return H


def _entropy_target_fidelity(
    mu_m: Float[torch.Tensor, "batch 1"],
    sigma_m: Float[torch.Tensor, "batch 1"],
    f_star: Float[torch.Tensor, "N"],
) -> Float[torch.Tensor, "batch 1"]:
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
    mu_m: Float[torch.Tensor, "batch 1"],
    sigma_m: Float[torch.Tensor, "batch 1"],
    f_star: Float[torch.Tensor, "N"],
    model: BARKMOGP,
    x: Shaped[torch.Tensor, "batch D"],
    fidelity_vector: Int[torch.Tensor, "batch 1"],
) -> Float[torch.Tensor, "1 1"]:
    # define fidelity vectors
    target_fidelity_vector = torch.tensor([[0]])
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

    def Psi(f: Float[torch.Tensor, "G 1 1"]) -> Float[torch.Tensor, "G batch N"]:
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

    # adaptive integral range to include full support of Psi

    init_min_integration = -10.0
    init_max_integration = 10.0
    num_adaptive_locations = 100
    adaptive_bound = 0.25

    f_range_adapt = torch.linspace(
        init_min_integration, init_max_integration, steps=num_adaptive_locations
    )
    psi = Psi(f_range_adapt.reshape(-1, 1, 1))
    psi_nonzeros = psi.abs().sum(dim=-1).sum(dim=-1) > 1e-8
    min_integration = f_range_adapt[psi_nonzeros].min() - adaptive_bound
    max_integration = f_range_adapt[psi_nonzeros].max() + adaptive_bound

    if min_integration < init_min_integration or max_integration > init_max_integration:
        warnings.warn(
            "Integration does not cover Psi; consider increasing initial range."
        )

    num_of_integration_steps = 250
    f_range = torch.linspace(
        min_integration, max_integration, steps=num_of_integration_steps
    )
    # y_vals has shape (num_of_integration_steps, batch_size, f_star.shape[0])
    z_phi = Z * Psi(f_range.reshape(-1, 1, 1))
    integral_grid = torch.special.xlogy(z_phi, z_phi)
    # estimate integral using trapezium rule
    integral_estimates = torch.trapezoid(integral_grid, f_range, dim=0)
    # now estimate H2 using Monte Carlo
    H_2 = -integral_estimates.mean(dim=-1).reshape(-1, 1)
    return H_2
