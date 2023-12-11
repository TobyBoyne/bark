import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.alternating.af_kernel import ATGP, AFGP
from alfalfa.alternating.fitting import fit_tree_gp
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.benchmarks import branin

torch.manual_seed(42)
N_train = 50
x = torch.rand((N_train, 2)) * 15 + torch.tensor([-5, 0])
y = branin(x)


likelihood = gpy.likelihoods.GaussianLikelihood()
gp = AFGP(x, y, likelihood)

# mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)
mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
# Calc loss and backprop gradients
loss = -mll(output, y)
print(f"Initial loss={loss}")

fit_tree_gp(x, y, gp, likelihood, mll)

output = gp(x)
loss = -mll(output, y)
print(f"Final loss={loss}")
gp.eval()

test_x = torch.meshgrid(torch.linspace(-5, 10, 50), torch.linspace(0, 15, 50), indexing="ij")
plot_gp_2d(gp, likelihood, x, y, test_x, target=branin)
plt.show()
