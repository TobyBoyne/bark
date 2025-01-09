import argparse
import logging
import pathlib
from typing import TypedDict

import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import RandomStrategy
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate

import bark.utils.metrics as metrics
from bark.benchmarks import map_benchmark
from bark.bofire_utils.data_models.strategies.mapper import strategy_map
from bark.bofire_utils.data_models.surrogates.api import BARKSurrogate, LeafGPSurrogate
from bark.bofire_utils.data_models.surrogates.mapper import surrogate_map

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


def _get_surrogate_datamodel(config: Config, domain: Domain):
    if config["model"] == "GP":
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

    raise KeyError(f"Model {config['model']} not found")


def main(seed: int, config: Config):
    benchmark = map_benchmark(config["benchmark"], **config.get("benchmark_params", {}))
    domain = benchmark.domain

    # sample initial points
    sampler = strategy_map(RandomStrategy(domain=domain, seed=seed))

    surrogate_dm = _get_surrogate_datamodel(config, domain)
    surrogate = surrogate_map(surrogate_dm)

    logger.info(f"Sample train data (n={config['num_train']})")
    train_x = sampler.ask(config["num_train"])
    experiments = benchmark.f(train_x, return_complete=True)

    logger.info("Tell experiments and fit surrogate")
    surrogate.fit(experiments)

    logger.info(f"Sample test data (n={config['num_test']})")
    test_x = sampler.ask(config["num_test"])
    test_experiments = benchmark.f(test_x, return_complete=True)

    logger.info("Predict")
    test_predictions = surrogate.predict(test_experiments)

    nlpd = metrics.nlpd(
        test_predictions["y_pred"].to_numpy(),
        test_predictions["y_sd"].to_numpy() ** 2,
        test_experiments["y"].to_numpy(),
    )
    logger.info(f"NLPD = {nlpd}")
    return test_predictions


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
