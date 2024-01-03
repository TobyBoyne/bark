import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.tree_models.forest import AlfalfaForest
from alfalfa.tree_models.tree_kernels import AFGP
from alfalfa.fitting import alternating_fit
from alfalfa.utils.plots import plot_gp_1d, plot_covar_matrix


def func(x):
    return (5 * x * torch.sin(5 * x)).flatten()

torch.manual_seed(42)
x = torch.tensor([0, 0.5, 0.9]).reshape(-1, 1)
# x = torch.tensor([0, 0.1, 0.2, 0.3, 0.9]).reshape(-1, 1)
# x = torch.tensor([0, 0.1, 0.11, 0.12, 0.13, 0.14, 0.2, 0.3, 0.9]).reshape(-1, 1)
x = torch.tensor([0, 0.02, 0.1, 0.11, 0.12, 0.13, 0.14, 0.2, 0.3, 0.7, 0.9]).reshape(-1, 1)
x = torch.rand((10, 1))
y = func(x)

test_x = torch.linspace(0, 1, 500).reshape((-1, 1))
test_y = func(test_x)

likelihood = gpy.likelihoods.GaussianLikelihood()
forest = AlfalfaForest(depth=3, num_trees=20)
# uncomment this line to show issue with complete trees
# forest = AlfalfaForest(depth=2, num_trees=1)
forest.initialise_forest([0])

gp = AFGP(x, y, likelihood, forest)

mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
# Calc loss and backprop gradients
loss = -mll(output, y)

alternating_fit(x, y, gp, mll, test_x, test_y)

# randomise the forest again
# forest.initialise_forest([0])
output = gp(x)
loss = -mll(output, y)
gp.eval()
plot_gp_1d(gp, likelihood, x, y, test_x, target=func)
plot_covar_matrix(gp, test_x)
plt.show()
