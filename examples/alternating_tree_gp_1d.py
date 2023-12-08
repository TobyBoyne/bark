import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.alternating.af_kernel import ATGP
from alfalfa.alternating.fitting import fit_tree_gp
from alfalfa.utils.plots import plot_gp_1d

torch.manual_seed(42)
x = torch.linspace(0, 1, 10).reshape((-1, 1))
y = (5 * x * torch.sin(5 * x)).flatten()


likelihood = gpy.likelihoods.GaussianLikelihood()
gp = ATGP(x, y, likelihood)

# gp.eval()
# x_test = torch.linspace(0, 5, 50).reshape((-1, 1))

mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
# Calc loss and backprop gradients
loss = -mll(output, y)
print(loss)
print(gp.tree.root.threshold)

fit_tree_gp(x, y, gp, likelihood, mll)

output = gp(x)
loss = -mll(output, y)
print("> ", loss)
gp.eval()
print(gp.tree)
plot_gp_1d(gp, likelihood, x, y)
plt.show()
