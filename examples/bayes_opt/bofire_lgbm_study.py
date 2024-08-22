import bofire.strategies.api as strategies
import gpytorch as gpy
import pandas as pd
import pydantic
import torch
from bofire.benchmarks.single import Hartmann
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy
from botorch import fit_gpytorch_mll

# with install_import_hook("alfalfa", "beartype.beartype"):
from alfalfa.bofire_utils.sampling import sample_projected
from alfalfa.fitting import fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.optimizer.opt_core import get_opt_core_from_domain
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.domain import get_feature_types_array

benchmark = Hartmann()
domain = benchmark.domain

# sample initial points
try:
    sampler = strategies.map(RandomStrategy(domain=domain, seed=44))
    train_x = sampler.ask(10)
except pydantic.ValidationError:
    train_x = sample_projected(domain, n=10, seed=44)
train_y = benchmark.f(train_x)["y"]  # .drop("valid_y", axis="columns")
cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)
transform_specs = {k: CategoricalEncodingEnum.ORDINAL for k in cat}
feature_types = get_feature_types_array(domain)

# add model_core with constraints if problem has constraints
model_core = get_opt_core_from_domain(domain)
# main bo loop
print("\n* * * start bo loop...")
for itr in range(100):
    train_x_transformed = domain.inputs.transform(train_x, transform_specs)
    train_y_transformed = (train_y - train_y.mean()) / train_y.std()
    booster = fit_lgbm_forest(train_x_transformed, train_y_transformed, domain)
    forest = lgbm_to_alfalfa_forest(booster)

    likelihood = gpy.likelihoods.GaussianLikelihood()
    train_torch = map(
        lambda x: torch.from_numpy(x.to_numpy()),
        (train_x_transformed, train_y_transformed),
    )
    tree_gp = AlfalfaGP(*train_torch, likelihood, forest)
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, tree_gp)
    fit_gpytorch_mll(mll)

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest, feature_types)
    opt_model = build_opt_model(
        benchmark.domain, gbm_model, tree_gp, 1.96, model_core=model_core
    )
    next_x = propose(benchmark.domain, opt_model, gbm_model, model_core)
    candidate = pd.DataFrame(data=[next_x], columns=domain.inputs.get_keys())
    candidate = domain.inputs.inverse_transform(candidate, transform_specs)
    next_y = benchmark.f(candidate)["y"]

    # update progress
    train_x = pd.concat((train_x, candidate), ignore_index=True)
    train_y = pd.concat((train_y, next_y), ignore_index=True)

    print(f"{itr}. min_val: {min(train_y):.5f}")

print(benchmark.f(train_x))
