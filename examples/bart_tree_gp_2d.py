import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from alfalfa.fitting import BART, BARTData, BARTTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.bb_funcs import Branin
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.space import Space

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)
N_train = 50
x = torch.rand((N_train, 2))
bb_func = Branin()
f = bb_func.vector_apply(x)

y = f + torch.randn_like(f) * 0.2**0.5


likelihood = gpy.likelihoods.GaussianLikelihood(
    noise_constraint=gpy.constraints.Positive()
)
forest = AlfalfaForest(height=0, num_trees=10)
space = Space([[0.0, 1.0], [0.0, 1.0]])
forest.initialise(space)
gp = AlfalfaGP(x, y, likelihood, forest)


mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
loss = -mll(output, y)
print(f"Initial loss={loss}")

test_x = torch.rand((50, 2))
test_f = bb_func.vector_apply(test_x)
test_y = test_f + torch.randn_like(test_f) * 0.2**0.5

data = BARTData(space, np.asarray(x))
params = BARTTrainParams(
    warmup_steps=10,
    n_steps=10,
    lag=500 // 5,  # want 5 samples
)
bart = BART(
    gp,
    data,
    params,
    scale_prior=stats.gamma(3.0, scale=1.94 / 3.0),
    noise_prior=stats.gamma(3.0, scale=0.057 / 3.0),
)
logger = bart.run()

output = gp(x)
loss = -mll(output, y)
print(f"Final loss={loss}")
gp.eval()

sampled_model = AlfalfaGP.from_mcmc_samples(gp, logger["samples"])
sampled_model.eval()


torch.save(sampled_model.state_dict(), "models/branin_sampled_bart_.pt")
test_x = torch.meshgrid(
    torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij"
)
plot_gp_2d(sampled_model, test_x, target=bb_func.vector_apply)
fig, axs = plt.subplots(nrows=2)
axs[0].plot(logger["noise"])
axs[1].plot(logger["scale"])
plt.show()
