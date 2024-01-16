from alfalfa import AlfalfaTree, AlfalfaForest
from alfalfa.tree_models.tree_kernels import ATGP, AFGP
from alfalfa.tree_models.forest import Node
from alfalfa.fitting import mcmc_fit, MCMCTrainParams

import math
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from matplotlib import pyplot as plt


# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 4)
train_x = torch.tensor([0.0, 0.1, 0.3, 0.9])
# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
# Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
forest = AlfalfaForest(depth=2, num_trees=2)
forest.initialise_forest([0])
model = AFGP(train_x, train_y, likelihood, forest)

num_samples = 5
mcmc_fit(train_x, train_y, model, 
         train_params=MCMCTrainParams(warmup_steps=num_samples, num_samples=num_samples))

model.eval()
test_x = torch.linspace(0, 1, 101).unsqueeze(-1)
test_y = torch.sin(test_x * (2 * math.pi))
expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
output = model(expanded_test_x)


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*', zorder=10)

    for i in range(min(num_samples, 25)):
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), output.mean[i].detach().numpy(), 'b', linewidth=0.3)

    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Sampled Means'])

plt.show()