import argparse
import logging
import pathlib

import numpy as np
import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import RandomStrategy
from typing_extensions import NotRequired, TypedDict

from bark.benchmarks import DatasetBenchmark, map_benchmark
from bark.bofire_utils.data_models.strategies.mapper import strategy_map
from bark.bofire_utils.data_models.surrogates.api import (
    BARKSurrogate,
)
from bark.bofire_utils.data_models.surrogates.mapper import surrogate_map
from bark.bofire_utils.domain import get_feature_types_array
from bark.fitting.bark_sampler import DataT, ModelT
from bark.tree_kernels.tree_gps import batched_forest_gram_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

NUM_RUNS = 1


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


def mll(model: ModelT, data: DataT, domain: Domain):
    forest, noise, scale = model
    train_x, y = data
    n = train_x.shape[0]

    feat_types = get_feature_types_array(domain)
    forest = forest.reshape(-1, *forest.shape[-2:])
    noise = noise.reshape(-1)
    K_XX = batched_forest_gram_matrix(forest, train_x, train_x, feat_types)
    # K_XX is (batch x n x n)
    K_XX_s = K_XX + (1e-6 + noise[:, None, None]) * np.eye(n)

    K_inv = np.linalg.inv(K_XX_s)
    _, K_logdet = np.linalg.slogdet(K_XX_s)
    y = y[None, ...]
    data_fit = (y.transpose((0, 2, 1)) @ K_inv @ y).squeeze()
    mll_arr = 0.5 * (-data_fit - K_logdet - n * np.log(2 * np.pi))
    return mll_arr


def main(seed: int, benchmark_config: BenchmarkConfig, model_config: ModelConfig):
    benchmark = map_benchmark(
        benchmark_config["benchmark"], **benchmark_config.get("benchmark_params", {})
    )
    domain = benchmark.domain

    # sample initial points
    seed_rng = np.random.default_rng(seed)
    all_mlls = np.zeros((NUM_RUNS, model_config["model_params"]["num_samples"]))
    for i, run_seed in enumerate(seed_rng.choice(2**32, size=NUM_RUNS, replace=False)):
        if isinstance(benchmark, DatasetBenchmark):
            benchmark._num_sampled = 0
            sampler_fn = lambda n_samples: benchmark.sample(n_samples, seed=run_seed)
        else:
            sampler = strategy_map(RandomStrategy(domain=domain, seed=run_seed))
            sampler_fn = sampler.ask

        surrogate_dm = _get_surrogate_datamodel(model_config, domain)
        surrogate = surrogate_map(surrogate_dm)

        logger.info(f"Sample train data (n={benchmark_config['num_train']})")
        train_x = sampler_fn(benchmark_config["num_train"])
        experiments = benchmark.f(train_x, return_complete=True)

        logger.info("Tell experiments and fit surrogate")
        surrogate.fit(experiments)

        logger.info("Compute MLL")
        mll_arr = mll(surrogate.model_as_tuple(), surrogate.train_data, domain)
        all_mlls[i, :] = mll_arr

    return all_mlls


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

    all_mlls = main(seed, benchmark_config, model_config)

    output_dir = (
        pathlib.Path(args.output_dir)
        / benchmark_config.get("benchmark_save_name", benchmark_config["benchmark"])
        / model_config.get("model_save_name", model_config["model"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "mlls.npy", all_mlls)

    config = {**benchmark_config, **model_config}
    yaml.dump(config, open(output_dir / "config.yaml", "w"))
