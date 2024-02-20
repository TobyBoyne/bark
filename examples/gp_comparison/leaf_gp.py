import gpytorch as gpy
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch

from alfalfa.fitting import fit_leaf_gp, lgbm_to_alfalfa_forest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.bb_funcs import get_func
from alfalfa.utils.plots import plot_gp_nd

torch.set_default_dtype(torch.float64)
plt.style.use(["science", "no-latex", "grid"])

bb_func = get_func("branin")

torch.manual_seed(42)
np.random.seed(42)

init_data = bb_func.get_init_data(30, rnd_seed=42)
space = bb_func.get_space()
X, y = init_data["X"], init_data["y"]

train_x, train_y = np.asarray(X), np.asarray(y)

tree_model = lgb.train(
    {"max_depth": 3, "min_data_in_leaf": 1},
    lgb.Dataset(train_x, train_y),
    num_boost_round=50,
)

forest = lgbm_to_alfalfa_forest(tree_model)
forest.initialise(space)
likelihood = gpy.likelihoods.GaussianLikelihood()

gp = AlfalfaGP(torch.from_numpy(train_x), torch.from_numpy(train_y), likelihood, forest)
fit_leaf_gp(gp)
gp.eval()
torch.save(gp.state_dict(), "models/branin_leaf_gp.pt")

test_x = torch.meshgrid(
    torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij"
)

plot_gp_nd(gp, test_x, target=bb_func.vector_apply)
plt.show()
