from ...tree_kernels import AlfalfaGP
from .data import Data
from .params import BARTTrainParams
import numpy as np
import scipy.stats as stats
import gpytorch as gpy

STEP_SIZE = 0.1

def softplus(x):
    """Used to transform from unconstrained space to constrained space"""
    return np.log(1 + np.exp(x))

def log_q_ratio_lognorm(cur_val, new_val):
    """Compute the log transition ratio for a lognormal proposal"""
    log_q_star = stats.lognorm.logpdf(cur_val, s=STEP_SIZE, scale=new_val)
    log_q = stats.lognorm.logpdf(new_val, s=STEP_SIZE, scale=cur_val)
    return log_q_star - log_q

def propose_noise_transition(model: AlfalfaGP):
    # take a proposal in the unconstrained space

    cur_raw_noise = model.likelihood.raw_noise.item()
    new_raw_noise = cur_raw_noise + np.random.randn() * STEP_SIZE
    new_noise = softplus(new_raw_noise)
    return new_noise

def noise_acceptance_probability(model: AlfalfaGP, new_noise: float, prior: stats.rv_continuous):
    cur_noise = model.likelihood.noise.item()
    
    log_q_ratio = log_q_ratio_lognorm(cur_noise, new_noise)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    output = model(model.train_inputs[0])
    likelihood = mll(output, model.train_targets)

    model.likelihood.noise = new_noise
    output = model(model.train_inputs[0])
    likelihood_star = mll(output, model.train_targets)

    likelihood_ratio = (likelihood_star - likelihood).item()

    # prior ratio
    prior_ratio = prior.logpdf(new_noise) - prior.logpdf(cur_noise)

    return min(log_q_ratio + likelihood_ratio + prior_ratio, 0.0)


def propose_scale_transition(model: AlfalfaGP):
    # take a proposal in the unconstrained space

    cur_raw_scale = model.covar_module.raw_outputscale.item()
    new_raw_scale = cur_raw_scale + np.random.randn() * STEP_SIZE
    new_scale = softplus(new_raw_scale)
    return new_scale

def scale_acceptance_probability(model: AlfalfaGP, new_scale: float, prior: stats.rv_continuous):
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
