from alfalfa import AlfalfaTree, AlfalfaForest
from alfalfa.tree_models.tree_kernels import ATGP, AFGP, AlfalfaGP
from alfalfa.tree_models.forest import Node
from dataclasses import dataclass
import math
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from matplotlib import pyplot as plt
from typing import Union, Optional

from torch.nn import Module as TModule
from torch.distributions import Bernoulli
from gpytorch.priors import UniformPrior, Prior

class BernoulliPrior(Prior, Bernoulli):
    """Bernoulli prior.
    """

    def __init__(self, probs, validate_args=False, transform=None):
        TModule.__init__(self)
        Bernoulli.__init__(self, probs=probs, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return BernoulliPrior(self.probs.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(Bernoulli, self).__call__(*args, **kwargs)

@dataclass
class MCMCTrainParams:
    num_samples: int = 100
    warmup_steps: int = 100


# Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.

def _apply_priors_to_tree_model(tree_model: Union[AlfalfaTree, AlfalfaForest]):
    if isinstance(tree_model, AlfalfaForest):
        for tree in tree_model.trees:
            _apply_priors_to_tree_model(tree)
            return

    # for node in tree_model.root.modules():
    #     if isinstance(node, Node) and not node.is_leaf:
    #         node.register_prior("is_leaf_prior", BernoulliPrior(0.1), "is_leaf")
    #         node.register_prior("threshold_prior", UniformPrior(0.0, 1.0), "threshold")

    # don't include the last layer, as this represents the maximum depth
    for d in range(tree_model.depth):
        for node in tree_model.nodes_by_depth[d]:
            node.register_prior("is_leaf_prior", BernoulliPrior(0.1), "is_leaf")
            node.register_prior("threshold_prior", UniformPrior(0.0, 1.0), "threshold")

def _apply_priors(model: AlfalfaGP):
    _apply_priors_to_tree_model(model.forest)

    model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    model.likelihood.register_prior("noise_prior", UniformPrior(0.01, 2.0), "noise")


def mcmc_fit(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        model: AlfalfaGP,
        train_params: Optional[MCMCTrainParams] = None
    ):

    if train_params is None:
        train_params = MCMCTrainParams()

    _apply_priors(model)

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(x))
            pyro.sample("obs", output, obs=y)
        return y

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(
        nuts_kernel,
        num_samples=train_params.num_samples, 
        warmup_steps=train_params.warmup_steps, 
        disable_progbar=False
    )
    mcmc_run.run(train_x, train_y)

    model.pyro_load_from_samples(mcmc_run.get_samples())
    return mcmc_run