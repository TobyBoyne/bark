import gpytorch
import numpy as np
import scienceplots  # noqa: F401
import scipy.stats as stats
import torch
from linear_operator.operators import DiagLinearOperator
from matplotlib import pyplot as plt

from alfalfa.benchmarks import map_benchmark
from alfalfa.fitting.bart.bart import BART
from alfalfa.fitting.bart.data import Data
from alfalfa.fitting.bart.params import BARTTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.tree_kernels import AlfalfaGP, AlfalfaSampledModel

torch.set_default_dtype(torch.float64)
plt.style.use(["science", "no-latex", "grid"])

bb_func = map_benchmark("branin")
# bb_func = map_benchmark("himmelblau1d")

# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
np.random.seed(42)

train_x, train_y = bb_func.get_init_data(30, rnd_seed=42)
space = bb_func.get_space()

tree = AlfalfaForest(height=0, num_trees=50)
data = Data(space, train_x)
tree.initialise(space, data.get_init_prior())
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.Positive()
)
model = AlfalfaGP(torch.tensor(train_x), torch.tensor(train_y), likelihood, tree)

N = 100
LAG = 10
params = BARTTrainParams(warmup_steps=100, n_steps=N, lag=LAG, alpha=0.95)
bart = BART(
    model,
    data,
    params,
    scale_prior=stats.halfnorm(scale=1.0),
    noise_prior=stats.halfnorm(scale=1.0),
)

logger = bart.run()

sampled_gp = AlfalfaSampledModel(
    torch.tensor(train_x), torch.tensor(train_y), logger["samples"], space, seed=42
)

# test_x, test_y = map(torch.as_tensor, bb_func.grid_sample((100,)))
test_x, test_y = map(torch.as_tensor, bb_func.grid_sample((25, 25)))

model.eval()
model_pd = model.likelihood(model(test_x))
model_pd = gpytorch.distributions.MultivariateNormal(
    model_pd.mean, DiagLinearOperator(model_pd.covariance_matrix.diag())
)

pred_dists = (("Samples", sampled_gp(test_x)), ("Latest", model_pd))
for model, pred_dist in pred_dists:
    print(
        f"""
          Model={model},
          NLPD={gpytorch.metrics.negative_log_predictive_density(pred_dist, test_y)}"""
    )

plt.show()
