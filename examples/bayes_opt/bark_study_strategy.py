import argparse
import logging
import pathlib
from typing import TypedDict

import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import (
    EntingStrategy,
    RandomStrategy,
    SoboStrategy,
)

from bark.benchmarks import map_benchmark
from bark.bofire_utils.data_models.strategies.api import TreeKernelStrategy
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


class Config(TypedDict):
    benchmark: str
    benchmark_params: dict
    num_init: int
    num_iter: int
    model: str
    model_params: dict


def _get_strategy_datamodel(config: Config, domain: Domain):
    if config["model"] == "Sobo":
        return SoboStrategy(domain=domain, seed=seed)
    if config["model"] == "BARK":
        return TreeKernelStrategy(
            domain=domain,
            seed=seed,
            surrogate_specs=BARKSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
                **config.get("model_params", {}),
            ),
        )
    if config["model"] == "LeafGP":
        return TreeKernelStrategy(
            domain=domain,
            seed=seed,
            surrogate_specs=LeafGPSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
                **config.get("model_params", {}),
            ),
        )
    if config["model"] == "Entmoot":
        return EntingStrategy(domain=domain, seed=seed)
    if config["model"] == "BARKPrior":
        return TreeKernelStrategy(
            domain=domain,
            seed=seed,
            surrogate_specs=BARKPriorSurrogate(
                inputs=domain.inputs,
                outputs=domain.outputs,
                **config.get("model_params", {}),
            ),
        )

    raise KeyError(f"Strategy {config['model']} not found")


def main(seed: int, config: Config):
    benchmark = map_benchmark(config["benchmark"], **config.get("benchmark_params", {}))
    domain = benchmark.domain

    # sample initial points
    sampler = strategy_map(RandomStrategy(domain=domain, seed=seed))
    train_x = sampler.ask(config["num_init"])
    experiments = benchmark.f(train_x, return_complete=True)

    strategy_dm = _get_strategy_datamodel(config, domain)
    strategy = strategy_map(strategy_dm)
    strategy.tell(experiments)

    logger.info("Start BO loop")
    for itr in range(config["num_iter"]):
        logger.info(f"Ask for datapoint {itr=}")
        candidate = strategy.ask(1)
        logger.info("Evaluate")
        experiment = benchmark.f(candidate, return_complete=True)
        logger.info("Tell")
        strategy.tell(experiment)

    return strategy.experiments


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--seed", type=str)
    argparser.add_argument("-c", "--config_file", type=str)
    argparser.add_argument("-o", "--output_dir", type=str)

    args = argparser.parse_args()
    seed = args.seed
    config = yaml.safe_load(open(args.config_file))

    experiments = main(seed, config)

    output_dir = pathlib.Path(args.output_dir) / config["benchmark"] / config["model"]
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments.to_csv(output_dir / f"seed={seed}.csv", index=False)
    yaml.dump(config, open(output_dir / "config.yaml", "w"))
