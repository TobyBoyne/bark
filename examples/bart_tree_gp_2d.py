import torch
import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np

from alfalfa.tree_models.tree_kernels import AlfalfaGP
from alfalfa.tree_models.forest import AlfalfaForest
from alfalfa.fitting import BART, BARTData, BARTTrainParams
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.benchmarks import rescaled_branin
from alfalfa.leaf_gp.space import Space

torch.manual_seed(42)
np.random.seed(42)
N_train = 50
x = torch.rand((N_train, 2)) 
f = rescaled_branin(x)

y = f + torch.randn_like(f) * 0.2**0.5


likelihood = gpy.likelihoods.GaussianLikelihood()
forest = AlfalfaForest(height=0, num_trees=5)
space = Space([[0.0, 1.0], [0.0, 1.0]])
forest.initialise(space)
gp = AlfalfaGP(x, y, likelihood, forest)

mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
loss = -mll(output, y)
print(f"Initial loss={loss}")

test_x = torch.rand((50, 2)) 
test_f = rescaled_branin(test_x)
test_y = test_f + torch.randn_like(test_f) * 0.2**0.5

data = BARTData(space, np.asarray(x))
params = BARTTrainParams(warmup_steps=500)
bart = BART(gp, data, params)
bart.run()

output = gp(x)
loss = -mll(output, y)
print(f"Final loss={loss}")
gp.eval()
torch.save(gp.state_dict(), "models/branin_bart.pt")
test_x = torch.meshgrid(torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij")
plot_gp_2d(gp, likelihood, x, y, test_x, target=rescaled_branin)
plt.show()
