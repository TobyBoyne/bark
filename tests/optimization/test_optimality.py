import numpy as np
import pytest
from bofire.benchmarks.api import Benchmark, Himmelblau
from bofire.data_models.domain.api import Domain

from bark.benchmarks import StyblinskiTang
from bark.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from bark.forest import batched_forest_gram_matrix, create_empty_forest
from bark.optimizer import propose
from bark.optimizer.opt_core import get_opt_core_from_domain
from bark.optimizer.opt_model import build_opt_model_from_forest
from bofire_mixed.domain import get_feature_types_array


def train_data(benchmark: Benchmark, n: int) -> tuple[np.ndarray, np.ndarray]:
    train_x = benchmark.domain.inputs.sample(n)
    train_y = benchmark.f(train_x)["y"]

    train_x_transformed = train_x.to_numpy()
    train_y_transformed = ((train_y - train_y.mean()) / train_y.std()).to_numpy()[
        :, None
    ]
    return (train_x_transformed, train_y_transformed)


def forest_predict(
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    data: tuple[np.ndarray, np.ndarray],
    candidates: np.ndarray,
    domain: Domain,
) -> np.ndarray:
    forest, noise, scale = model
    forest = forest.reshape(-1, *forest.shape[-2:])
    noise = noise.reshape(-1)
    scale = scale.reshape(-1)

    num_samples = scale.shape[0]
    num_candidates = candidates.shape[0]

    train_x, train_y = data
    feature_types = get_feature_types_array(domain)
    K_XX = scale[:, None, None] * batched_forest_gram_matrix(
        forest, train_x, train_x, feature_types
    )
    K_XX_s = K_XX + noise[:, None, None] * np.eye(train_x.shape[0])

    K_inv = np.linalg.inv(K_XX_s)
    K_xX = scale[:, None, None] * batched_forest_gram_matrix(
        forest, candidates, train_x, feature_types
    )

    mu = K_xX @ K_inv @ train_y
    var = scale[:, None, None] - K_xX @ K_inv @ K_xX.transpose((0, 2, 1))

    mu = mu.reshape(num_samples, num_candidates)
    var = np.diagonal(var, axis1=1, axis2=2)
    return mu, var


def calculate_acqf(mu: np.ndarray, var: np.ndarray, kappa: float) -> np.ndarray:
    std = np.sqrt(var)
    acqf = mu - kappa * std
    return acqf.mean(axis=0)


# @pytest.mark.slow
@pytest.mark.parametrize("benchmark", [Himmelblau(), StyblinskiTang(dim=10)])
def test_proposal_maximises_acqf(benchmark: Benchmark):
    domain = benchmark.domain
    data_numpy = train_data(benchmark, n=15)

    model_core = get_opt_core_from_domain(domain)

    bark_params = BARKTrainParams(
        warmup_steps=500, n_steps=400, thinning=200, num_chains=4
    )

    forest = create_empty_forest(m=50)
    forest = np.tile(forest, (bark_params.num_chains, 1, 1))
    noise = np.tile(0.1, (bark_params.num_chains,))
    scale = np.tile(1.0, (bark_params.num_chains,))

    samples = run_bark_sampler(
        model=(forest, noise, scale),
        data=data_numpy,
        domain=domain,
        params=bark_params,
    )

    opt_model = build_opt_model_from_forest(
        domain=benchmark.domain,
        gp_samples=samples,
        data=data_numpy,
        kappa=1.96,
        model_core=model_core,
    )

    next_x = propose(benchmark.domain, opt_model, model_core)
    next_x_candidate = np.array([next_x])
    candidates = domain.inputs.sample(n=1000, seed=42).to_numpy()

    mu, var = forest_predict(samples, data_numpy, candidates, domain)
    acqf = calculate_acqf(mu, var, kappa=1.96)

    mux, varx = forest_predict(samples, data_numpy, next_x_candidate, domain)
    acqfx = calculate_acqf(mux, varx, kappa=1.96)

    assert acqfx.item() <= acqf.min()
