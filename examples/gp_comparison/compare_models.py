import gpytorch as gpy
import matplotlib.pyplot as plt
import problem
import scienceplots  # noqa: F401
import torch

from bark.baselines import RBFGP
from bark.tree_kernels import AlfalfaGP
from bark.utils.plots import plot_gp_nd


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
    gp.tree_model.initialise(problem.bb_func.space)
    return gp


plt.style.use(["science", "no-latex", "grid"])


test_x = torch.meshgrid(
    torch.linspace(0, 1, 25), torch.linspace(0, 1, 25), indexing="ij"
)

test_X1, test_X2 = test_x
test_x_stacked = torch.stack((test_X1.flatten(), test_X2.flatten()), dim=1)
test_y = problem.bb_func.vector_apply(test_x_stacked)

models = (
    ("RBF", "models/branin_rbf_gp.pt", _get_rbf_gp),
    ("Leaf-GP", "models/branin_leaf_gp.pt", _get_forest_gp),
    ("BART", "models/branin_bart_gp.pt", _get_forest_gp),
)

for name, path, model_fn in models:
    model = model_fn(path, problem.train_x_torch, problem.train_y_torch)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    output = model(model.train_inputs[0])
    loss = -mll(output, model.train_targets)

    model.eval()
    fig, ax = plot_gp_nd(model, test_x, target=problem.bb_func.vector_apply)
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
