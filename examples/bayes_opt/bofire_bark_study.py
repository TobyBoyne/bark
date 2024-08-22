import bofire.strategies.api as strategies
import pandas as pd
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy

from alfalfa.benchmarks import StyblinskiTang
from alfalfa.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from alfalfa.forest_numba import create_empty_forest
from alfalfa.optimizer import propose
from alfalfa.optimizer.opt_core import get_opt_core_from_domain
from alfalfa.optimizer.opt_model import build_opt_model_from_forest
from alfalfa.utils.domain import get_feature_types_array

benchmark = StyblinskiTang()
domain = benchmark.domain

# sample initial points
sampler = strategies.map(RandomStrategy(domain=domain, seed=42))
train_x = sampler.ask(10)
train_y = benchmark.f(train_x)["y"]  # .drop("valid_y", axis="columns")
cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)
feature_types = get_feature_types_array(domain)

# add model_core with constraints if problem has constraints
model_core = get_opt_core_from_domain(domain)

# main bo loop
bark_params = BARKTrainParams(warmup_steps=500, n_steps=500, thinning=100)

forest = create_empty_forest(m=50)
noise = 0.1
scale = 1.0

print("\n* * * start bo loop...")
for itr in range(100):
    train_x_transformed = train_x.to_numpy()
    train_y_transformed = ((train_y - train_y.mean()) / train_y.std()).to_numpy()[
        :, None
    ]
    data_numpy = (train_x_transformed, train_y_transformed)
    samples = run_bark_sampler(
        model=(forest, noise, scale),
        data=data_numpy,
        domain=domain,
        params=bark_params,
    )

    bark_params.warmup_steps = 0

    # get new proposal and evaluate bb_func
    opt_model = build_opt_model_from_forest(
        domain=benchmark.domain,
        gp_samples=samples,
        data=data_numpy,
        kappa=1.96,
        model_core=model_core,
    )
    next_x = propose(benchmark.domain, opt_model, model_core)
    candidate = pd.DataFrame(data=[next_x], columns=domain.inputs.get_keys())
    next_y = benchmark.f(candidate)["y"]

    # update progress
    train_x = pd.concat((train_x, candidate), ignore_index=True)
    train_y = pd.concat((train_y, next_y), ignore_index=True)

    print(f"{itr}. min_val: {min(train_y):.5f}")
