import gpytorch as gpy
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import torch
from bofire.benchmarks.api import Himmelblau
from bofire.data_models.features.api import CategoricalInput

from alfalfa.fitting import fit_gp_adam, fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.metrics import nlpd

plt.style.use(["science", "no-latex", "grid"])


benchmark = Himmelblau()
train_x = benchmark.domain.inputs.sample(10)
train_y = benchmark.f(train_x).drop("valid_y", axis="columns")
cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)

params = {}
tree_model = fit_lgbm_forest(train_x, train_y, benchmark.domain, params)

forest = lgbm_to_alfalfa_forest(tree_model)
forest.initialise(benchmark.domain)
likelihood = gpy.likelihoods.GaussianLikelihood()

gp = AlfalfaGP(
    torch.from_numpy(train_x.to_numpy()),
    torch.from_numpy(train_y.to_numpy()).reshape(-1),
    likelihood,
    forest,
)
fit_gp_adam(gp)
gp.eval()

test_x = benchmark.domain.inputs.sample(100)
test_y = benchmark.f(test_x).drop("valid_y", axis="columns")


output = gp.likelihood(gp(torch.from_numpy(test_x.to_numpy())))
test_loss = nlpd(output, torch.from_numpy(test_y.to_numpy()).reshape(-1), diag=False)
print(f"GP test loss={test_loss}")

# torch.save(gp.state_dict(), "models/branin_leaf_gp.pt")

# test_x = torch.meshgrid(
#     torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing="ij"
# )

# plot_gp_nd(gp, test_x, target=problem.bb_func.vector_apply)
# plt.show()
