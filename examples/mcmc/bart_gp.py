import gpytorch
import numpy as np
import scienceplots  # noqa: F401
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from alfalfa.fitting.bart.bart import BART
from alfalfa.fitting.bart.data import Data
from alfalfa.fitting.bart.params import BARTTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.bb_funcs import get_func
from alfalfa.utils.plots import plot_gp_nd

torch.set_default_dtype(torch.float64)
plt.style.use(["science", "no-latex", "grid"])

bb_func = get_func("branin")
# bb_func = get_func("himmelblau1d")

# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
np.random.seed(42)

init_data = bb_func.get_init_data(30, rnd_seed=42)
space = bb_func.get_space()
X, y = init_data["X"], init_data["y"]

train_x, train_y = np.asarray(X), np.asarray(y)
# train_y = np.zeros_like(train_y) + np.random.randn(*train_y.shape) * 0.0

tree = AlfalfaForest(height=0, num_trees=50)
data = Data(space, train_x)
tree.initialise(space, data.get_init_prior())
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Positive()
)
model = AlfalfaGP(torch.tensor(train_x), torch.tensor(train_y), likelihood, tree)

N = 500
LAG = 5
params = BARTTrainParams(warmup_steps=0, n_steps=N, lag=LAG, alpha=0.95)
bart = BART(
    model,
    data,
    params,
    scale_prior=stats.halfnorm(scale=1.0),
    noise_prior=stats.halfnorm(scale=1.0),
)

logger = bart.run()
model.eval()

fig = plt.figure(figsize=(8, 4))
gs = GridSpec(2, 2, figure=fig)
ax_func = fig.add_subplot(gs[:, 0])
ax_hist = fig.add_subplot(gs[0, 1])
ax_hyper = fig.add_subplot(gs[1, 1])

noise = logger["noise"]
scale = logger["scale"]
mlls = logger["mll"]
samples = logger["samples"]

t = np.arange(0, N, LAG)
(l_n,) = ax_hyper.plot(t, noise, label="GP Noise")
(l_s,) = ax_hyper.plot(t, scale, label="Kernel Scale")
(l_mll,) = ax_hyper.plot(t, mlls, label="MLL")

ax_hist.hist(torch.tensor(noise), density=True, bins=20)
x_prior = np.linspace(0, 2, 50)
ax_hist.plot(x_prior, bart.noise_prior.pdf(x_prior), label="prior")
ax_hist.legend()
# print(sum(logger["accept_noise"]) / len(logger["accept_noise"]))
ax_hyper.legend()

ax_hyper.set_xlim(0, N)
ax_hyper.set_ylim(0, 2.5)
ax_hyper.set_xlabel("Iteration #")

# GP
idx = np.argmin(mlls)

forest_dict = samples[idx]["covar_module.base_kernel._extra_state"]["tree_model"]
forest = AlfalfaForest.from_dict(forest_dict)
forest.initialise(space)
model = AlfalfaGP(torch.tensor(train_x), torch.tensor(train_y), likelihood, forest)
model.eval()
model.likelihood.noise = noise[idx]
model.covar_module.outputscale = scale[idx]

torch.save(model.state_dict(), "models/branin_bart_gp.pt")

# test_x = torch.linspace(0, 1, steps=50).reshape(-1, 1)
test_x = torch.meshgrid(
    torch.linspace(0, 1, 25), torch.linspace(0, 1, 25), indexing="ij"
)

plot_gp_nd(model, test_x, bb_func.vector_apply, ax_func)

ax_func.set_ylim(-2.5, 2.5)
ax_func.set_xlim(0.0, 1.0)

plt.show()
