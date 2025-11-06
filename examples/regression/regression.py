import logging
from pathlib import Path
from time import perf_counter
from typing import Annotated

import jax
import numpy as np
import pandas as pd
import typer
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import RandomStrategy
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.strategies.api import map as strategy_map
from typer import Option

import bark.utils.metrics as metrics
from bark.utils import script_utils
from bofire_mixed.benchmarks import DatasetBenchmark, map_benchmark
from bofire_mixed.data_models.surrogates.api import (
    BARKSurrogate,
    BARTSurrogate,
    LeafGPSurrogate,
)
from bofire_mixed.data_models.surrogates.mapper import surrogate_map

jax.config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

NUM_RUNS = 20


def _get_surrogate_datamodel(model_config: script_utils.ModelConfig, domain: Domain):
    model_params = model_config.model_params
    model_name = model_config.model
    if model_name == "GP":
        return SingleTaskGPSurrogate(inputs=domain.inputs, outputs=domain.outputs)
    if model_name == "BARK":
        return BARKSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **model_params,
        )
    if model_name == "LeafGP":
        return LeafGPSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **model_params,
        )
    if model_name == "BART":
        return BARTSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            **model_params,
        )

    raise KeyError(f"Model {model_name} not found")


def run_experiments(
    seed: int,
    benchmark_config: script_utils.BenchmarkConfig,
    model_config: script_utils.ModelConfig,
) -> pd.DataFrame:
    benchmark = map_benchmark(
        benchmark_config.benchmark, **benchmark_config.benchmark_params
    )
    domain = benchmark.domain

    # sample initial points
    seed_rng = np.random.default_rng(seed)
    all_metrics = []
    for run_seed in seed_rng.choice(2**32, size=NUM_RUNS, replace=False):
        if isinstance(benchmark, DatasetBenchmark):
            benchmark._num_sampled = 0
            sampler_fn = lambda n_samples: benchmark.sample(n_samples, seed=run_seed)
        else:
            sampler = strategy_map(RandomStrategy(domain=domain, seed=run_seed))
            sampler_fn = sampler.ask

        surrogate_dm = _get_surrogate_datamodel(model_config, domain)
        surrogate = surrogate_map(surrogate_dm)

        logger.info(f"Sample train data (n={benchmark_config.num_train})")
        train_x = sampler_fn(benchmark_config.num_train)
        experiments = benchmark.f(train_x, return_complete=True)

        logger.info("Tell experiments and fit surrogate")
        start_time = perf_counter()
        surrogate.fit(experiments)
        time_taken = perf_counter() - start_time

        logger.info(f"Sample test data (n={benchmark_config.num_test})")
        test_x = sampler_fn(benchmark_config.num_test)
        test_experiments = benchmark.f(test_x, return_complete=True)

        logger.info("Predict")
        test_predictions = surrogate.predict(test_experiments)

        y_lbl = domain.outputs.get_keys()[0]
        y_pred_lbl, y_sd_lbl = f"{y_lbl}_pred", f"{y_lbl}_sd"
        nlpd = metrics.nlpd(
            test_predictions[y_pred_lbl].to_numpy(),
            test_predictions[y_sd_lbl].to_numpy() ** 2,
            test_experiments[y_lbl].to_numpy(),
        )
        mse = metrics.mse(
            test_predictions[y_pred_lbl].to_numpy(), test_experiments[y_lbl].to_numpy()
        )
        logger.info(f"NLPD = {nlpd},\t MSE = {mse}")
        all_metrics.append([nlpd, mse, time_taken])

    return pd.DataFrame(data=all_metrics, columns=["NLPD", "MSE", "Time"])


def main(
    seed: Annotated[int, Option("--seed", "-s")],
    benchmark_config_file: Annotated[Path, Option("--benchmark", "-b")],
    model_config_file: Annotated[Path, Option("--model", "-m")],
    output_dir: Annotated[Path, Option("--output", "-o")],
):
    benchmark_config = script_utils.read_benchmark_config(benchmark_config_file)
    model_config = script_utils.read_model_config(model_config_file)

    experiments = run_experiments(seed, benchmark_config, model_config)

    experiments_output_dir = script_utils.get_output_path(
        output_dir, benchmark_config, model_config, write_config=True
    )

    experiments.to_csv(experiments_output_dir / f"seed={seed}.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
