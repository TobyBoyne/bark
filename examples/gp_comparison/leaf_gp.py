import gpytorch as gpy
import lightgbm as lgb
import matplotlib.pyplot as plt
import problem
import scienceplots  # noqa: F401

from alfalfa.fitting import fit_gp_adam, lgbm_to_alfalfa_forest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.metrics import nlpd

plt.style.use(["science", "no-latex", "grid"])


space = problem.bb_func.space
tree_model = lgb.train(
    {"max_depth": 3, "min_data_in_leaf": 1},
    lgb.Dataset(problem.train_x_np, problem.train_y_np),
    num_boost_round=50,
)

forest = lgbm_to_alfalfa_forest(tree_model)
forest.initialise(space)
likelihood = gpy.likelihoods.GaussianLikelihood()

gp = AlfalfaGP(problem.train_x_torch, problem.train_y_torch, likelihood, forest)
fit_gp_adam(gp)
gp.eval()

output = gp.likelihood(gp(problem.test_x_torch))
test_loss = nlpd(output, problem.test_y_torch, diag=False)
print(f"GP test loss={test_loss}")

# torch.save(gp.state_dict(), "models/branin_leaf_gp.pt")

# test_x = torch.meshgrid(
#     torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij"
# )

# plot_gp_nd(gp, test_x, target=problem.bb_func.vector_apply)
# plt.show()
