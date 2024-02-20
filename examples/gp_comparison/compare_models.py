import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots # noqa: F401

from alfalfa.baselines import RBFGP
from alfalfa.utils.space import Space
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.bb_funcs import get_func
from alfalfa.utils.plots import plot_gp_nd


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


torch.set_default_dtype(torch.float64)
plt.style.use(["science", "no-latex", "grid"])



torch.manual_seed(42)
np.random.seed(42)
bb_func = get_func("branin")

init_data = bb_func.get_init_data(30, rnd_seed=42)
space = bb_func.get_space()
X, y = init_data["X"], init_data["y"]

train_x, train_y = np.asarray(X), np.asarray(y)

test_x = torch.meshgrid(
    torch.linspace(0, 1, 25), torch.linspace(0, 1, 25), indexing="ij"
)

test_X1, test_X2 = test_x
test_x_stacked = torch.stack((test_X1.flatten(), test_X2.flatten()), dim=1)
test_y = bb_func.vector_apply(test_x_stacked)

models = (
    ("RBF", "models/branin_rbf_gp.pt", _get_rbf_gp),
    ("Leaf-GP", "models/branin_leaf_gp.pt", _get_forest_gp),
    ("BART", "models/branin_bart_gp.pt", _get_forest_gp),
)

for name, path, model_fn in models:
    model = model_fn(path, torch.from_numpy(train_x), torch.from_numpy(train_y))
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    output = model(model.train_inputs[0])
    loss = -mll(output, model.train_targets)

    model.eval()
    fig, ax = plot_gp_nd(model, test_x, target=bb_func.vector_apply)
    fig.suptitle(f"{name} Model")

    pred_dist = model.likelihood(model(test_x_stacked))

    print(
        f"""Model {name}:
    MLL= {loss:.3f}
    NLPD={gpy.metrics.negative_log_predictive_density(pred_dist, test_y):.4f}"""
    )

# fig, ax = plt.subplots()

# ax.contourf(test_X1, test_X2, test_y.reshape(test_X1.shape))

# plt.show()
