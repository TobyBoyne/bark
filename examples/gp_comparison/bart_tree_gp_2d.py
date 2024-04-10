import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from jaxtyping import install_import_hook

with install_import_hook("alfalfa", "beartype.beartype"):
    from alfalfa.benchmarks import Branin
    from alfalfa.fitting import BART, BARTData, BARTTrainParams
    from alfalfa.forest import AlfalfaForest
    from alfalfa.tree_kernels import AlfalfaGP


torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)

bb_func = Branin(seed=42)
x, f = bb_func.get_init_data(50, rnd_seed=42)
x = torch.as_tensor(x)
f = torch.as_tensor(f)
y = f + torch.randn_like(f) * 0.2**0.5

likelihood = gpy.likelihoods.GaussianLikelihood(
    noise_constraint=gpy.constraints.Positive()
)
forest = AlfalfaForest(height=0, num_trees=20)
space = bb_func.space
forest.initialise(space)
gp = AlfalfaGP(x, y, likelihood, forest)


mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

output = gp(x)
loss = -mll(output, y)
print(f"Initial loss={loss}")

data = BARTData(space, np.asarray(x))
params = BARTTrainParams(
    warmup_steps=100,
    n_steps=10,
    lag=500 // 5,  # want 5 samples
)
bart = BART(
    gp,
    data,
    params,
    scale_prior=stats.halfnorm(scale=1.0),
    noise_prior=stats.halfnorm(scale=1.0),
)
logger = bart.run()

output = gp(x)
loss = -mll(output, y)
print(f"Final loss={loss}")
gp.eval()

sampled_model = AlfalfaGP.from_mcmc_samples(gp, logger["samples"])
sampled_model.eval()


# torch.save(sampled_model.state_dict(), "models/branin_sampled_bart_.pt")
# test_x_shape = np.full(len(space), fill_value=50)
# test_x, _ = bb_func.grid_sample(test_x_shape)
# plot_gp_nd(sampled_model, torch.as_tensor(test_x), target=bb_func.vector_apply)

fig, axs = plt.subplots(nrows=2)
axs[0].plot(logger["noise"])
axs[1].plot(logger["scale"])
plt.show()
