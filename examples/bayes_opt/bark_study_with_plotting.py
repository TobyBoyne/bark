import argparse
import pathlib
from typing import TypedDict

import bofire.strategies.api as strategies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy

from bark.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from bark.forest import create_empty_forest
from bark.optimizer.opt_core import get_opt_core_from_domain
from bark.optimizer.opt_model import build_opt_model_from_forest
from bark.optimizer.proposals import propose
from bark.tree_kernels.tree_gps import forest_predict
from bofire_mixed.benchmarks import map_benchmark


class Config(TypedDict):
    benchmark: str
    benchmark_params: dict
    num_init: int
    num_iter: int
    model: str
    model_params: dict


def main(seed: int, config: Config):
    benchmark = map_benchmark(config["benchmark"], **config.get("benchmark_params", {}))
    domain = benchmark.domain

    # sample initial points
    sampler = strategies.map(RandomStrategy(domain=domain, seed=seed))
    train_x = sampler.ask(config["num_init"])
    train_y = benchmark.f(train_x)["y"]  # .drop("valid_y", axis="columns")
    cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)
    transform_specs = {k: CategoricalEncodingEnum.ORDINAL for k in cat}

    # add model_core with constraints if problem has constraints
    model_core = get_opt_core_from_domain(domain)

    bark_params = BARKTrainParams(**config["model_params"])

    forest = create_empty_forest(m=50)
    forest = np.tile(forest, (bark_params.num_chains, 1, 1))
    noise = np.tile(0.1, (bark_params.num_chains,))
    scale = np.tile(1.0, (bark_params.num_chains,))

    print("\n* * * start bo loop...")
    for itr in range(config["num_iter"]):
        train_x_transformed = domain.inputs.transform(train_x, transform_specs)
        train_x_transformed = train_x_transformed.to_numpy()
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
        print("Training complete")

        # get new proposal and evaluate bb_func
        opt_model = build_opt_model_from_forest(
            domain=benchmark.domain,
            gp_samples=samples,
            data=data_numpy,
            kappa=1.96,
            model_core=model_core,
        )

        # update starting values
        forest = samples[0][:, -1, :, :]
        noise = samples[1][:, -1]
        scale = samples[2][:, -1]

        next_x = propose(benchmark.domain, opt_model, model_core)
        candidate = pd.DataFrame(data=[next_x], columns=domain.inputs.get_keys())
        candidate_inv_transform = domain.inputs.inverse_transform(
            candidate, transform_specs
        )
        next_y = benchmark.f(candidate_inv_transform)["y"]

        test_x = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        test_x = np.array([t.flatten() for t in test_x]).T
        test_x_df = pd.DataFrame(data=test_x, columns=domain.inputs.get_keys())
        test_y = benchmark.f(test_x_df)["y"].to_numpy()

        mu, var = forest_predict(samples, data_numpy, test_x, domain)
        pred_y = mu.mean(axis=0)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

        ax[0].imshow(test_y.reshape(100, 100), extent=(0, 1, 0, 1), origin="lower")
        ax[1].imshow(pred_y.reshape(100, 100), extent=(0, 1, 0, 1), origin="lower")

        ax[0].scatter(train_x_transformed[:, 0], train_x_transformed[:, 1])

        plt.show()

        # update progress
        train_x = pd.concat((train_x, candidate_inv_transform), ignore_index=True)
        train_y = pd.concat((train_y, next_y), ignore_index=True)

        print(f"{itr}. min_val: {min(train_y):.5f}")

    return train_x, train_y


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--seed", type=str)
    argparser.add_argument("-c", "--config_file", type=str)
    argparser.add_argument("-o", "--output_dir", type=str)

    args = argparser.parse_args()
    seed = args.seed
    config = yaml.safe_load(open(args.config_file))

    x, y = main(seed, config)

    output_dir = pathlib.Path(args.output_dir) / config["benchmark"] / config["model"]
    output_dir.mkdir(parents=True, exist_ok=True)
    x.to_csv(output_dir / f"seed={seed}_x.csv", index=False)
    y.to_csv(output_dir / f"seed={seed}_y.csv", index=False)
    yaml.dump(config, open(output_dir / "config.yaml", "w"))
