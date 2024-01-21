import torch
import gpytorch as gpy
import matplotlib.pyplot as plt

from alfalfa.tree_models.tree_kernels import ATGP, AFGP
from alfalfa.tree_models.forest import AlfalfaForest, prune_tree_hook
from alfalfa.gps import RBFGP
from alfalfa.utils.plots import plot_gp_2d
from alfalfa.utils.benchmarks import rescaled_branin

def _get_rbf_gp(path, x, y):
    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = RBFGP(x, y, likelihood)
    gp.load_state_dict(torch.load(path))
    return gp

def _get_forest_gp(path, x, y):
    state = torch.load(path)
    forest_state = state["covar_module.base_kernel.forest._extra_state"]
    likelihood = gpy.likelihoods.GaussianLikelihood()
    forest = AlfalfaForest(depth=forest_state["depth"], num_trees=forest_state["num_trees"])
    forest.initialise_forest([0, 0], randomise=False)
    gp = AFGP(x, y, likelihood, forest)
    gp.register_load_state_dict_post_hook(prune_tree_hook)
    gp.load_state_dict(torch.load(path))
    return gp

torch.manual_seed(42)
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
    ("AF", "models/branin_alternating_forest_.pt", _get_forest_gp),
)

for name, path, model_fn in models:
    model = model_fn(path, x, y)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    output = model(x)
    loss = -mll(output, y)

    model.eval()
    fig, ax = plot_gp_2d(model, model.likelihood, x, y, test_x, target=rescaled_branin)
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