import argparse
import logging
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import RandomStrategy
from bofire.data_models.surrogates.api import BotorchSurrogate

import bark.utils.metrics as metrics
from bark.benchmarks import map_benchmark
from bark.bofire_utils.data_models.api import BARKSurrogate
from bark.bofire_utils.data_models.mapper import strategy_map, surrogate_map

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
    if config["model"] == "Sobo":
        return BotorchSurrogate(inputs=domain.inputs, outputs=domain.outputs)
    elif config["model"] == "BARK":
        return BARKSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **config.get("model_params", {}),
        )
    else:
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
    all_nlpds = np.zeros((surrogate_dm.num_chains, surrogate_dm.num_samples))
    full_forest, full_noise, full_scale = surrogate.model_as_tuple()
    for i in range(surrogate_dm.num_chains):
        for j in range(surrogate_dm.num_samples):
            surrogate.forest = full_forest[: i + 1, : j + 1, :, :]
            surrogate.noise = full_noise[: i + 1, : j + 1]
            surrogate.scale = full_scale[: i + 1, : j + 1]
            test_predictions = surrogate.predict(test_experiments)

            nlpd = metrics.nlpd(
                test_predictions["y_pred"].to_numpy(),
                test_predictions["y_sd"].to_numpy() ** 2,
                test_experiments["y"].to_numpy(),
            )
            logger.info(f"NLPD({i}, {j}) = {nlpd}")
            all_nlpds[i, j] = nlpd
    return all_nlpds


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--seed", type=str)
    argparser.add_argument("-c", "--config_file", type=str)
    argparser.add_argument("-o", "--output_dir", type=str)

    args = argparser.parse_args()
    seed = args.seed
    config = yaml.safe_load(open(args.config_file))

    all_nlpds = main(seed, config)

    fig, ax = plt.subplots()
    img = ax.imshow(all_nlpds, cmap="viridis")
    fig.colorbar(img)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Chains")
    fig.savefig("figs/samples_ablation_m25.pdf")
    plt.show()
