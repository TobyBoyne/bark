import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.tree_models.tree_kernels import ATGP, AFGP
from alfalfa.tree_models.forest import AlfalfaForest
from alfalfa.fitting import alternating_fit
from alfalfa.utils.plots import plot_gp_2d

torch.manual_seed(42)
N_train = 50
x0 = torch.rand((N_train,))
x1 = torch.randint(0, 3, (N_train,))

x = torch.stack((x0, x1), dim=1)
f = torch.where(x1 > 1, x0, 10 * x0 - 2)
y = f + torch.randn_like(f) * 0.2**0.5


likelihood = gpy.likelihoods.GaussianLikelihood()
forest = AlfalfaForest(depth=2, num_trees=10)
forest.initialise_forest(var_is_cat=[0, 3])
gp = AFGP(x, y, likelihood, forest)

mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
# Calc loss and backprop gradients
loss = -mll(output, y)
print(f"Initial loss={loss}")

alternating_fit(x, y, gp, mll)

output = gp(x)
loss = -mll(output, y)
print(f"Final loss={loss}")
gp.eval()
torch.save(gp.state_dict(), "models/cat_alternating_forest.pt")
plt.show()
