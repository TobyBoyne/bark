import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.tree_models.tree_kernels import ATGP, AFGP
from alfalfa.tree_models.forest import AlfalfaForest
from alfalfa.tree_models.alternating_fitting import fit_tree_gp
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.benchmarks import rescaled_branin

torch.manual_seed(42)
N_train = 50
x = torch.rand((N_train, 2)) 
f = rescaled_branin(x)

y = f + torch.randn_like(f) * 0.2**0.5


likelihood = gpy.likelihoods.GaussianLikelihood()
forest = AlfalfaForest(depth=2, num_trees=10)
gp = AFGP(x, y, likelihood, forest)

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
torch.save(gp.state_dict(), "models/branin_alternating_forest_new.pt")
test_x = torch.meshgrid(torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij")
plot_gp_2d(gp, likelihood, x, y, test_x, target=rescaled_branin)
plt.show()
