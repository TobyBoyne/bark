import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.alternating.af_kernel import ATGP, AFGP
from alfalfa.alternating.alternating_forest import AlternatingForest
from alfalfa.alternating.fitting import fit_tree_gp
from alfalfa.gps import RBFGP
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.benchmarks import rescaled_branin

def _get_rbf_gp(path, x, y):
    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = RBFGP(x, y, likelihood)
    gp.load_state_dict(torch.load(path))
    return gp

def _get_af_gp(path, x, y):
    state = torch.load(path)
    depth = state["forest.trees.0._extra_state"]["depth"]
    likelihood = gpy.likelihoods.GaussianLikelihood()
    forest = AlternatingForest(depth=depth, num_trees=10)
    gp = AFGP(x, y, likelihood, forest)
    gp.load_state_dict(torch.load(path))
    return gp

torch.manual_seed(42)
N_train = 50
x = torch.rand((N_train, 2))
f = rescaled_branin(x)
sigma_noise = 0.2
y = f + torch.randn_like(f) * sigma_noise

models = (
    ("RBF", "models/branin_rbf_gp.pt", _get_rbf_gp),
    ("AF", "models/branin_alternating_forest.pt", _get_af_gp),
)

for name, path, model_fn in models:
    model = model_fn(path, x, y)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    output = model(x)
    loss = -mll(output, y)
    print(f"{name} loss={loss:.3f}")