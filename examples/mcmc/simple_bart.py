from alfalfa import AlfalfaTree, AlfalfaForest
from alfalfa.tree_models.tree_kernels import AlfalfaGP
from alfalfa.tree_models.forest import DecisionNode
from alfalfa.fitting.bart.bart import BART
from alfalfa.fitting.bart.data import Data
from alfalfa.leaf_gp.space import Space, Dimension
from alfalfa.fitting.bart.params import BARTTrainParams
from alfalfa.utils.plots import plot_gp_1d, plot_covar_matrix

import math
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
import scipy.stats as stats

# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 10).reshape(-1, 1)
space = Space([[0.0, 1.0]])

# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
np.random.seed(42)
f = lambda x: torch.sin(x * (2 * math.pi))
train_y = (f(train_x) + torch.randn(train_x.size()) * 0.2).flatten()

tree = AlfalfaForest(height=0, num_trees=3)
data = Data(space, train_x)
tree.initialise(space, data.get_init_prior())
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
model = AlfalfaGP(train_x, train_y, likelihood, tree)

params = BARTTrainParams(warmup_steps=10_000, lag=5)
bart = BART(model, data, params, 
            scale_prior=stats.halfnorm(scale=100.0),
            noise_prior=stats.halfnorm(scale=100.0)
)
logger = bart.run()

model.eval()
# torch.save(model.state_dict(), "models/1d_bart.pt")

test_x = torch.linspace(0, 1, 100).reshape(-1, 1)
fig, ax = plot_gp_1d(model, test_x, f)
fig, axs = plt.subplots(ncols=3)
axs[0].plot(logger["noise"])
axs[0].set_title("noise")
axs[1].plot(logger["scale"])
axs[1].set_title("scale")

with torch.no_grad():
    cov = model.covar_module(test_x).evaluate().numpy()
axs[2].imshow(cov, interpolation="nearest")

plt.show()