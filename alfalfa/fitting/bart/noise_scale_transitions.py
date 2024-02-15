import gpytorch as gpy
import numpy as np
import scipy.stats as stats

from ...tree_kernels import AlfalfaGP

STEP_SIZE = 0.1

def softplus(x):
    """Used to transform from unconstrained space to constrained space"""
    return np.log(1 + np.exp(x))


def log_q_ratio_lognorm(cur_val, new_val):
    """Compute the log transition ratio for a lognormal proposal"""
    log_q_star = stats.lognorm.logpdf(cur_val, s=STEP_SIZE, scale=new_val)
    log_q = stats.lognorm.logpdf(new_val, s=STEP_SIZE, scale=cur_val)
    return log_q_star - log_q


def propose_positive_transition(cur_value: float, step_size: float=STEP_SIZE) -> float:
    """Propose a new value for a hyperparameter that is positive.

    Proposals are made in the unconstrained log-space

    Args:
        cur_value (float): current value of hyperparameter
        step_size (float): size of proposed step in log-space

    Returns:
        float: proposed value
    """
    cur_log_value = np.log(cur_value + 1e-30)
    new_log_value = cur_log_value + np.random.randn() * step_size
    new_value = np.exp(new_log_value)
    return new_value


def noise_acceptance_probability(
    model: AlfalfaGP, new_noise: float, prior: stats.rv_continuous
):
    cur_noise = model.likelihood.noise.item()

    log_q_ratio = log_q_ratio_lognorm(cur_noise, new_noise)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    output = model(model.train_inputs[0])
    likelihood = mll(output, model.train_targets)

    model.likelihood.noise = new_noise
    output = model(model.train_inputs[0])
    likelihood_star = mll(output, model.train_targets)

    # undo temporary transition
    model.likelihood.noise = cur_noise

    likelihood_ratio = (likelihood_star - likelihood).item()
    # prior ratio
    prior_ratio = prior.logpdf(new_noise) - prior.logpdf(cur_noise)
    # likelihood_ratio = 0.0
    return min(log_q_ratio + likelihood_ratio + prior_ratio, 0.0)


def scale_acceptance_probability(
    model: AlfalfaGP, new_scale: float, prior: stats.rv_continuous
):
    cur_scale = model.covar_module.outputscale

    log_q_ratio = log_q_ratio_lognorm(cur_scale, new_scale)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    output = model(model.train_inputs[0])
    likelihood = mll(output, model.train_targets)

    model.covar_module.outputscale = new_scale
    output = model(model.train_inputs[0])
    likelihood_star = mll(output, model.train_targets)

    likelihood_ratio = (likelihood_star - likelihood).item()

    # prior ratio
    prior_ratio = prior.logpdf(new_scale) - prior.logpdf(cur_scale)

    return min(log_q_ratio + likelihood_ratio + prior_ratio, 0.0)
