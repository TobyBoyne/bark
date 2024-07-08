import gpytorch as gpy
import numpy as np
import torch
from bofire.benchmarks.api import Himmelblau
from bofire.data_models.features.api import CategoricalInput

from alfalfa.fitting import fit_gp_adam, fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.optimizer.optimizer_utils import get_opt_core_from_domain
from alfalfa.tree_kernels import AlfalfaGP

benchmark = Himmelblau()
train_x = benchmark.domain.inputs.sample(10, seed=42)
train_y = benchmark.f(train_x).drop("valid_y", axis="columns")
cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)

# add model_core with constraints if problem has constraints
model_core = get_opt_core_from_domain(benchmark.domain)

# main bo loop
print("\n* * * start bo loop...")
for itr in range(100):
    booster = fit_lgbm_forest(X_train, y_train)
    forest = lgbm_to_alfalfa_forest(booster)
    forest.initialise(benchmark.domain)
    likelihood = gpy.likelihoods.GaussianLikelihood()
    tree_gp = AlfalfaGP(
        torch.from_numpy(X_train), torch.from_numpy(y_train), likelihood, forest
    )
    fit_gp_adam(tree_gp)

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(
        benchmark.domain, gbm_model, tree_gp, 1.96, model_core=model_core
    )
    next_x = propose(benchmark.domain, opt_model, gbm_model, model_core)
    next_y = benchmark.f(next_x)

    # update progress
    X_train = np.concatenate((X_train, [next_x]))
    y_train = np.concatenate((y_train, [next_y]))

    print(f"{itr}. min_val: {round(min(y_train), 5)}")
