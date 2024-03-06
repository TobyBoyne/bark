from collections import namedtuple

import gpytorch
import pytest
import torch
from jaxtyping import install_import_hook

from alfalfa.benchmarks import Himmelblau1D
from alfalfa.fitting import fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.tree_kernels import AlfalfaGP

torch.set_default_dtype(torch.float64)

TrainingData = namedtuple("TrainingData", ["train_x", "train_y", "space"])

with install_import_hook("alfalfa", "beartype.beartype"):
    import alfalfa  # noqa: F401


@pytest.fixture(scope="session")
def training_data():
    bb_func = Himmelblau1D()
    data = bb_func.get_init_data(num_init=10, rnd_seed=42)
    return (*data, bb_func.get_space())


@pytest.fixture(scope="session")
def lgbm_model(training_data: TrainingData):
    train_x, train_y, space = training_data
    booster = fit_lgbm_forest(train_x, train_y)
    forest = lgbm_to_alfalfa_forest(booster)
    forest.initialise(space)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = AlfalfaGP(
        torch.from_numpy(train_x), torch.from_numpy(train_y), likelihood, forest
    )

    return model
