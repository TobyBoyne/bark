"""Run regression while varying hyperparameters of BARK"""

import argparse
import logging
import pathlib

import numpy as np
import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import RandomStrategy
from typing_extensions import NotRequired, TypedDict

import bark.utils.metrics as metrics
from bark.benchmarks import DatasetBenchmark, map_benchmark
from bark.bofire_utils.data_models.strategies.mapper import strategy_map
from bark.bofire_utils.data_models.surrogates.api import (
    BARKSurrogate,
)
from bark.bofire_utils.data_models.surrogates.mapper import surrogate_map

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class BenchmarkConfig(TypedDict):
    benchmark: str
    benchmark_save_name: NotRequired[str]
    benchmark_params: NotRequired[dict]
    num_train: int
    num_test: int


class ModelConfig(TypedDict):
    model: str
    model_save_name: NotRequired[str]
    model_params: NotRequired[dict]


def _get_surrogate_datamodel(model_config: ModelConfig, domain: Domain):
    model_params = model_config.get("model_params", {})
    model_name = model_config["model"]
    if model_name == "BARK":
        return BARKSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **model_params,
        )
    raise KeyError(f"Model {model_name} not found")


def main(seed: int, benchmark_config: BenchmarkConfig, model_config: ModelConfig):
    benchmark = map_benchmark(
        benchmark_config["benchmark"], **benchmark_config.get("benchmark_params", {})
    )
    domain = benchmark.domain

    # sample initial points
    model_params = model_config["model_params"]

    if isinstance(benchmark, DatasetBenchmark):
        benchmark._num_sampled = 0
        sampler_fn = lambda n_samples: benchmark.sample(n_samples, seed=seed)
    else:
        sampler = strategy_map(RandomStrategy(domain=domain, seed=seed))
        sampler_fn = sampler.ask

    logger.info(
        f"Sample train (n={benchmark_config['num_train']}) and test (n={benchmark_config['num_test']}) data"
    )
    train_x = sampler_fn(benchmark_config["num_train"])
    experiments = benchmark.f(train_x, return_complete=True)

    logger.info("Tell experiments and fit surrogate")

    logger.info(f"Sample test data (n={benchmark_config['num_test']})")
    test_x = sampler_fn(benchmark_config["num_test"])
    test_experiments = benchmark.f(test_x, return_complete=True)

    alpha_lst = model_params["alpha"]
    beta_lst = model_params["beta"]
    nlpd_arr = np.zeros((len(alpha_lst), len(beta_lst)))

    for i, alpha in enumerate(alpha_lst):
        for j, beta in enumerate(beta_lst):
            model_params["alpha"] = alpha
            model_params["beta"] = beta

            surrogate_dm = _get_surrogate_datamodel(model_config, domain)
            surrogate = surrogate_map(surrogate_dm)

            surrogate.fit(experiments)

            logger.info("Predict")
            test_predictions = surrogate.predict(test_experiments)

            y_lbl = domain.outputs.get_keys()[0]
            y_pred_lbl, y_sd_lbl = f"{y_lbl}_pred", f"{y_lbl}_sd"
            nlpd = metrics.nlpd(
                test_predictions[y_pred_lbl].to_numpy(),
                test_predictions[y_sd_lbl].to_numpy() ** 2,
                test_experiments[y_lbl].to_numpy(),
            )
            logger.info(f"NLPD = {nlpd}")
            nlpd_arr[i, j] = nlpd

    return nlpd_arr


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--seed", type=int)
    argparser.add_argument("-c", "--config_file_benchmark", type=str)
    argparser.add_argument("-m", "--config_file_model", type=str)
    argparser.add_argument("-o", "--output_dir", type=str)

    args = argparser.parse_args()
    seed = args.seed
    benchmark_config: BenchmarkConfig = yaml.safe_load(open(args.config_file_benchmark))
    model_config: ModelConfig = yaml.safe_load(open(args.config_file_model))

    experiments = main(seed, benchmark_config, model_config)

    output_dir = (
        pathlib.Path(args.output_dir)
        / benchmark_config.get("benchmark_save_name", benchmark_config["benchmark"])
        / model_config.get("model_save_name", model_config["model"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"nlpd_seed={seed}.npy", experiments)

    config = {**benchmark_config, **model_config}
    yaml.dump(config, open(output_dir / "config.yaml", "w"))
