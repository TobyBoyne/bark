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

from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior


@dataclass
class MCMCTrainParams:
    num_samples: int = 100
    warmup_steps: int = 100


# Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
# likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())

def _apply_priors_to_tree_model(tree_model: Union[AlfalfaTree, AlfalfaForest]):
    if isinstance(tree_model, AlfalfaForest):
        for tree in tree_model.trees:
            _apply_priors_to_tree_model(tree)
            return

    for node in tree_model.root.modules():
        if isinstance(node, Node) and not node.is_leaf:
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

# model.eval()
# test_x = torch.linspace(0, 1, 101).unsqueeze(-1)
# test_y = torch.sin(test_x * (2 * math.pi))
# expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
# output = model(expanded_test_x)


# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))

#     # Plot training data as black stars
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*', zorder=10)

#     for i in range(min(num_samples, 25)):
#         # Plot predictive means as blue line
#         ax.plot(test_x.numpy(), output.mean[i].detach().numpy(), 'b', linewidth=0.3)

#     # Shade between the lower and upper confidence bounds
#     # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Sampled Means'])

# plt.show()