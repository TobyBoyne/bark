import argparse
import logging
import pathlib

import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import (
    EntingStrategy,
    RandomStrategy,
    SoboStrategy,
)
from typing_extensions import NotRequired, TypedDict

from bark.benchmarks import map_benchmark
from bark.bofire_utils.data_models.strategies.api import (
    RelaxedSoboStrategy,
    SMACStrategy,
    TreeKernelStrategy,
)
from bark.bofire_utils.data_models.strategies.mapper import strategy_map
from bark.bofire_utils.data_models.surrogates.api import (
    BARKPriorSurrogate,
    BARKSurrogate,
    LeafGPSurrogate,
)

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


def _get_strategy_datamodel(model_config: ModelConfig, domain: Domain):
    model_params = model_config.get("model_params", {})
    model_name = model_config["model"]
    if model_name == "Sobo":
        if model_params.get("cont_relax", False):
            return RelaxedSoboStrategy(domain=domain, seed=seed)
        return SoboStrategy(domain=domain, seed=seed)
    if model_name == "BARK":
        return TreeKernelStrategy(
            domain=domain,
            seed=seed,
            surrogate_specs=BARKSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
                **model_params,
            ),
        )
    if model_name == "LeafGP":
        return TreeKernelStrategy(
            domain=domain,
            seed=seed,
            surrogate_specs=LeafGPSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
                **model_params,
            ),
        )
    if model_name == "Entmoot":
        return EntingStrategy(domain=domain, seed=seed)
    if model_name == "BARKPrior":
        return TreeKernelStrategy(
            domain=domain,
            seed=seed,
            surrogate_specs=BARKPriorSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
                **model_params,
            ),
        )
    if model_name == "SMAC":
        return SMACStrategy(
            domain=domain,
            seed=seed,
        )
    if model_name == "Random":
        return RandomStrategy(domain=domain, seed=seed)
    raise KeyError(f"Strategy {model_name} not found")


def main(seed: int, benchmark_config: BenchmarkConfig, model_config: ModelConfig):
    benchmark = map_benchmark(
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

    sampler = strategy_map(RandomStrategy(domain=domain, seed=seed))
    train_x = sampler.ask(benchmark_config["num_init"])
    experiments = benchmark.f(train_x, return_complete=True)

    strategy_dm = _get_strategy_datamodel(model_config, domain)
    strategy = strategy_map(strategy_dm)
    strategy.tell(experiments)

    logger.info("Start BO loop")
    for itr in range(benchmark_config["num_iter"]):
        logger.info(f"Ask for datapoint {itr=}")
        candidate = strategy.ask(1)
        logger.info("Evaluate")
        experiment = benchmark.f(candidate, return_complete=True)
        logger.info("Tell")
        strategy.tell(experiment)

    return strategy.experiments


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
