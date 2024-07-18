import bofire.strategies.api as strategies
import gpytorch as gpy
import pandas as pd
import torch
from bofire.benchmarks.detergent import Detergent
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy

from alfalfa.benchmarks.mixed_bofire import PressureVessel
from alfalfa.fitting import fit_gp_adam, fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.optimizer.optimizer_utils import get_opt_core_from_domain
from alfalfa.tree_kernels import AlfalfaGP

benchmark = Detergent()
benchmark = PressureVessel()
domain = benchmark.domain

# sample initial points
sampler = strategies.map(RandomStrategy(domain=domain, seed=42))
train_x = sampler.ask(10)
train_y = benchmark.f(train_x)["y1"]  # .drop("valid_y", axis="columns")
cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)

# add model_core with constraints if problem has constraints
model_core = get_opt_core_from_domain(domain)
# main bo loop
print("\n* * * start bo loop...")
for itr in range(10):
    booster = fit_lgbm_forest(train_x, train_y, domain)
    forest = lgbm_to_alfalfa_forest(booster)
    forest.initialise(domain)
    likelihood = gpy.likelihoods.GaussianLikelihood()
    train_torch = map(lambda x: torch.from_numpy(x.to_numpy()), (train_x, train_y))
    tree_gp = AlfalfaGP(*train_torch, likelihood, forest)
    fit_gp_adam(tree_gp)

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(
        benchmark.domain, gbm_model, tree_gp, 1.96, model_core=model_core
    )
    next_x = propose(benchmark.domain, opt_model, gbm_model, model_core)
    candidate = pd.DataFrame(data=[next_x], columns=domain.inputs.get_keys())
    next_y = benchmark.f(candidate)["y1"]

    # update progress
    train_x = pd.concat((train_x, candidate), ignore_index=True)
    train_y = pd.concat((train_y, next_y), ignore_index=True)

    print(f"{itr}. min_val: {min(train_y):.5f}")

print(benchmark.f(train_x))
