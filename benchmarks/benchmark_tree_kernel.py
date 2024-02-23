"""Benchmark the speed of kernel operations.

Justify use of linear-ops"""
import gpytorch as gpy
import pytest
import torch

from alfalfa.fitting import fit_lgbm_forest, lgbm_to_alfalfa_forest
from alfalfa.tree_kernels import AlfalfaGP
from alfalfa.utils.space import Space


@pytest.fixture
def model() -> AlfalfaGP:
    torch.random.manual_seed(42)
    train_x = torch.rand(10).reshape(-1, 1)
    train_f = torch.sin(train_x.flatten() * (2 * torch.pi))
    train_y = train_f + torch.randn(train_f.size()) * 0.1

    lgbm_tree = fit_lgbm_forest(train_x.numpy(), train_y.numpy())
    forest = lgbm_to_alfalfa_forest(lgbm_tree)
    forest.initialise(Space([[0.0, 1.0]]))

    likelihood = gpy.likelihoods.GaussianLikelihood()
    model = AlfalfaGP(train_x, train_y, likelihood, forest)
    return model


def test_evaluate_gram_matrix(benchmark, model: AlfalfaGP):
    train_x = model.train_inputs[0]

    def evaluate_gram_matrix():
        model.covar_module(train_x, train_x)

    benchmark(evaluate_gram_matrix)


def test_evaluate_mll(benchmark, model: AlfalfaGP):
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    def evaluate_mll():
        output = model(train_x)
        _ = -mll(output, train_y)

    benchmark(evaluate_mll)
