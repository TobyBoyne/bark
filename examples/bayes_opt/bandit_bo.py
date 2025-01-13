import argparse
import inspect
import logging
import pathlib
import warnings

import numpy as np
import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from typing_extensions import NotRequired, TypedDict

from bark.benchmarks import map_benchmark
from bark.benchmarks.MAX_bandit import MAXBandit
from bark.bofire_utils.data_models.surrogates.api import (
    BARKSurrogate,
    BARTSurrogate,
    LeafGPSurrogate,
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
    num_init: NotRequired[int]
    num_iter: NotRequired[int]


class ModelConfig(TypedDict):
    model: str
    model_save_name: NotRequired[str]
    model_params: NotRequired[dict]


def _get_surrogate_datamodel(config: ModelConfig, domain: Domain):
    if config["model"] == "Sobo":
        return SingleTaskGPSurrogate(inputs=domain.inputs, outputs=domain.outputs)
    if config["model"] == "BARK":
        return BARKSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **config.get("model_params", {}),
        )
    if config["model"] == "LeafGP":
        return LeafGPSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **config.get("model_params", {}),
        )
    if config["model"] == "BART":
        return BARTSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **config.get("model_params", {}),
        )

    raise KeyError(f"Model {config['model']} not found")


def ucb(mu: np.ndarray, std: np.ndarray, minimize=True):
    # mu, std are ([batch,] n, 1)
    acqf = (-mu if minimize else mu) + 1.96 * std
    if acqf.ndim == 3:
        acqf = acqf.mean(axis=0)

    return acqf


def main(seed: int, benchmark_config: BenchmarkConfig, model_config: ModelConfig):
    benchmark: MAXBandit = map_benchmark(
        benchmark_config["benchmark"], **benchmark_config.get("benchmark_params", {})
    )
    domain = benchmark.domain

    if "num_init" not in benchmark_config:
        benchmark_config["num_init"] = min(30, 2 * len(domain.inputs))

    if "num_iter" not in benchmark_config:
        benchmark_config["num_iter"] = 100

    # sample initial points
    logger.info(
        f"Benchmark: {benchmark_config['benchmark']}\nModel: {model_config['model']}"
    )
    logger.info(f"Sample {benchmark_config['num_init']} initial points")

    sampler = np.random.default_rng(seed)
    experiment_idxs = sampler.choice(
        len(benchmark.data), size=benchmark_config["num_init"], replace=False
    )
    experiments = benchmark.f_by_idx(experiment_idxs)

    surrogate_dm = _get_surrogate_datamodel(model_config, domain)
    surrogate = surrogate_map(surrogate_dm)
    surrogate.fit(experiments)

    logger.info("Start BO loop")
    for itr in range(benchmark_config["num_iter"]):
        remaining_idxs = np.array(
            [i for i in range(len(benchmark.data)) if i not in experiment_idxs]
        )
        logger.info(f"Ask for datapoint {itr=}")
        remaining_candidates = benchmark.f_by_idx(remaining_idxs)

        # we manually do the input transform here to access _predict for batch mode
        Xt = surrogate.inputs.transform(
            remaining_candidates, surrogate.input_preprocessing_specs
        )
        if "batched" in inspect.signature(surrogate._predict).parameters:
            mu, stds = surrogate._predict(Xt, batched=True)
        else:
            mu, stds = surrogate._predict(Xt)

        acqf = ucb(mu, stds)
        new_idx = remaining_idxs[np.argmax(acqf)]

        logger.info("Tell")
        experiment_idxs = np.concatenate((experiment_idxs, [new_idx]))
        experiments = benchmark.f_by_idx(experiment_idxs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate.fit(experiments)

    return experiments


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
    experiments.to_csv(output_dir / f"seed={seed}.csv", index=False)

    config = {**benchmark_config, **model_config}
    yaml.dump(config, open(output_dir / "config.yaml", "w"))
