import bofire.strategies.api as strategies
import gpytorch as gpy
import pandas as pd
import torch
from bofire.benchmarks.api import Himmelblau
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy

from alfalfa.fitting import BARK, BARKData, BARKTrainParams
from alfalfa.forest import AlfalfaForest
from alfalfa.optimizer import build_opt_model, propose
from alfalfa.optimizer.gbm_model import GbmModel
from alfalfa.optimizer.opt_core import get_opt_core_from_domain
from alfalfa.tree_kernels import AlfalfaGP

benchmark = Himmelblau()
domain = benchmark.domain

# sample initial points
sampler = strategies.map(RandomStrategy(domain=domain, seed=42))
train_x = sampler.ask(10)
train_y = benchmark.f(train_x)["y"]  # .drop("valid_y", axis="columns")
cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)

# add model_core with constraints if problem has constraints
model_core = get_opt_core_from_domain(domain)

# main bo loop

bark_data = BARKData(domain, train_x.values)
bark_params = BARKTrainParams(warmup_steps=50)

print("\n* * * start bo loop...")
for itr in range(10):
    forest = AlfalfaForest(height=0, num_trees=50)
    forest.initialise(domain)
    likelihood = gpy.likelihoods.GaussianLikelihood()
    train_torch = map(lambda x: torch.from_numpy(x.to_numpy()), (train_x, train_y))
    tree_gp = AlfalfaGP(*train_torch, likelihood, forest)

    bark_sampler = BARK(
        model=tree_gp,
        data=bark_data,
        params=bark_params,
    )

    bark_sampler.run()
    bark_params.warmup_steps = 50

    # get new proposal and evaluate bb_func
    gbm_model = GbmModel(forest)
    opt_model = build_opt_model(
        benchmark.domain, gbm_model, tree_gp, 1.96, model_core=model_core
    )
    next_x = propose(benchmark.domain, opt_model, gbm_model, model_core)
    candidate = pd.DataFrame(data=[next_x], columns=domain.inputs.get_keys())
    next_y = benchmark.f(candidate)["y"]

    # update progress
    train_x = pd.concat((train_x, candidate), ignore_index=True)
    train_y = pd.concat((train_y, next_y), ignore_index=True)

    print(f"{itr}. min_val: {min(train_y):.5f}")
