## Delete this
from collections import namedtuple

import gpytorch
import torch

from bark.benchmarks import Himmelblau1D
from bark.fitting import fit_lgbm_forest, lgbm_to_bark_forest
from bark.optimizer.nystrom import construct_nystrom_features, nystrom_samples
from bark.tree_kernels import BARKGP

TrainingData = namedtuple("TrainingData", ["train_x", "train_y", "space"])
torch.set_default_dtype(torch.float64)


def training_data():
    bb_func = Himmelblau1D()
    data = bb_func.get_init_data(num_init=10, rnd_seed=42)
    return (*data, bb_func.get_space())


def lgbm_model(training_data: TrainingData):
    train_x, train_y, space = training_data
    booster = fit_lgbm_forest(train_x, train_y)
    forest = lgbm_to_bark_forest(booster)
    forest.initialise(space)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = BARKGP(
        torch.from_numpy(train_x), torch.from_numpy(train_y), likelihood, forest
    )

    return model


###


def test_nystrom_sampling_gp(lgbm_model):
    (xs,) = lgbm_model.train_inputs
    x_samples = xs
    z = construct_nystrom_features(lgbm_model, x_samples)
    linear_model = z(xs)
    samples = nystrom_samples(linear_model, 10)
    print(samples.shape)
    assert False


test_nystrom_sampling_gp(lgbm_model(training_data()))
