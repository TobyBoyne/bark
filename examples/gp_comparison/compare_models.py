import torch
import gpytorch as gpy
import numpy as np
import matplotlib.pyplot as plt

from alfalfa.tree_models.tree_kernels import AlfalfaGP
from alfalfa.tree_models.forest import AlfalfaForest
from alfalfa.gps import RBFGP
from alfalfa.leaf_gp.space import Space
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.benchmarks import rescaled_branin

def _get_rbf_gp(path, x, y):
    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = RBFGP(x, y, likelihood)
    gp.load_state_dict(torch.load(path))
    return gp

def _get_forest_gp(path, x, y):
    state = torch.load(path)
    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = AlfalfaGP(x, y, likelihood, None)
    gp.load_state_dict(state)
    gp.tree_model.initialise(Space([[0.0, 1.0], [0.0, 1.0]]))
    return gp

torch.manual_seed(42)
np.random.seed(42)
N_train = 50
x = torch.rand((N_train, 2))
f = rescaled_branin(x)
noise_var = 0.2
y = f + torch.randn_like(f) * noise_var ** 0.5

test_x = torch.meshgrid(torch.linspace(0, 1, 25), torch.linspace(0, 1, 25), indexing="ij")

test_X1, test_X2 = test_x
test_y = rescaled_branin(torch.stack((test_X1.flatten(), test_X2.flatten()), dim=1))


models = (
    ("RBF", "models/branin_rbf_gp.pt", _get_rbf_gp),
    ("Leaf-GP", "models/branin_leaf_gp.pt", _get_forest_gp),
    ("BART", "models/branin_bart.pt", _get_forest_gp),
)

for name, path, model_fn in models:
    model = model_fn(path, x, y)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    output = model(x)
    loss = -mll(output, y)

    model.eval()
    fig, ax = plot_gp_2d(model, test_x, target=rescaled_branin)
    fig.suptitle(f"{name} Model")

    test_X1, test_X2 = test_x
    test_x_stacked = torch.stack((test_X1.flatten(), test_X2.flatten()), dim=1)
    test_y = rescaled_branin(test_x_stacked)
    pred_dist = model.likelihood(model(test_x_stacked))

    print(f"""Model {name}:
    MLL= {loss:.3f}
    NLPD={gpy.metrics.negative_log_predictive_density(pred_dist, test_y):.4f}"""
    )

fig, ax = plt.subplots()

ax.contourf(test_X1, test_X2, test_y.reshape(test_X1.shape))

plt.show()