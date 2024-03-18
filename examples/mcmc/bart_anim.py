import gpytorch
import numpy as np
import scienceplots  # noqa: F401
import scipy.stats as stats
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from alfalfa.benchmarks import map_benchmark
from alfalfa.fitting.bart.bart import BART
from alfalfa.fitting.bart.data import Data
from alfalfa.fitting.bart.params import BARTTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.plots import plot_forest, plot_gp_1d

torch.set_default_dtype(torch.float64)
plt.style.use(["science", "no-latex", "grid"])

bb_func = map_benchmark("himmelblau1d")

# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
np.random.seed(42)

init_data = bb_func.get_init_data(10, rnd_seed=42)
space = bb_func.get_space()
X, y = init_data

train_x, train_y = np.asarray(X), np.asarray(y)

tree = AlfalfaForest(height=0, num_trees=1)
data = Data(space, train_x)
tree.initialise(space, data.get_init_prior())
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Positive()
)
model = AlfalfaGP(torch.tensor(train_x), torch.tensor(train_y), likelihood, tree)

N = 1000
LAG = 5
params = BARTTrainParams(warmup_steps=0, n_steps=N, lag=LAG, alpha=0.7)
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
# identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
ax_tree = fig.add_subplot(gs[0, 1])
ax_hyper = fig.add_subplot(gs[1, 1])

noise = logger["noise"]
scale = logger["scale"]
mlls = logger["mll"]
samples = logger["samples"]
forests = [
    AlfalfaForest.from_dict(s["covar_module.base_kernel._extra_state"]["tree_model"])
    for s in samples
]

for forest in forests:
    forest.initialise(space)

t = np.arange(0, N, LAG)
(l_n,) = ax_hyper.plot(noise[0], label="GP Noise")
(l_s,) = ax_hyper.plot(scale[0], label="Kernel Scale")
(l_mll,) = ax_hyper.plot(mlls[0], label="MLL")

ax_hyper.legend()

ax_hyper.set_xlim(0, N)
ax_hyper.set_ylim(0, 2.5)
ax_hyper.set_xlabel("Iteration #")
test_x = torch.linspace(0, 1, steps=50).reshape(-1, 1)


def update_fig(f):
    frame = int(f * (f / (N // LAG)) ** 1)
    # hypers
    l_n.set_data(t[:frame], noise[:frame])
    l_s.set_data(t[:frame], scale[:frame])
    l_mll.set_data(t[:frame], mlls[:frame])

    # forest
    ax_tree.clear()
    plot_forest(forests[frame], ax_tree)
    ax_tree.set_ylim(-10, 2)
    ax_tree.set_xlim(-4, 24)
    ax_tree.grid(False)
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])

    # gp
    ax_func.clear()
    model = AlfalfaGP(
        torch.tensor(train_x), torch.tensor(train_y), likelihood, forests[frame]
    )
    model.eval()
    model.likelihood.noise = noise[frame]
    model.covar_module.outputscale = scale[frame]
    plot_gp_1d(model, test_x, bb_func.vector_apply, ax_func)

    ax_func.set_ylim(-2.5, 2.5)
    ax_func.set_xlim(0.0, 1.0)

    return l_n, l_s


ani = FuncAnimation(fig, func=update_fig, frames=N // LAG, interval=60)
ani.save("figs/bart_anim_mll.mp4")
