from collections import namedtuple

import gpytorch
import pytest
import torch

torch.set_default_dtype(torch.float64)

TrainingData = namedtuple("TrainingData", ["train_x", "train_y", "space"])


@pytest.fixture(scope="session")
def training_data():
    from bark.benchmarks import Himmelblau1D

    bb_func = Himmelblau1D()
    data = bb_func.get_init_data(num_init=10, rnd_seed=42)
    return (*data, bb_func.get_space())


@pytest.fixture(scope="session")
def lgbm_model(training_data: TrainingData):
    from bark.fitting import fit_lgbm_forest, lgbm_to_bark_forest
    from bark.tree_kernels import BARKGP

    train_x, train_y, space = training_data
    booster = fit_lgbm_forest(train_x, train_y)
    forest = lgbm_to_bark_forest(booster)
    forest.initialise(space)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BARKGP(
        torch.from_numpy(train_x), torch.from_numpy(train_y), likelihood, forest
    )

    return model
