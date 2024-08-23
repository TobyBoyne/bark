"""Demonstration of information based MF"""

import numpy as np
import torch

from bark.benchmarks import CurrinExp2D
from bark.fitting import fit_gp_adam, fit_lgbm_forest, lgbm_to_alfalfa_forest
from bark.optimizer.information_based_fidelity import (
    propose_fidelity_information_based,
)
from bark.tree_kernels import AlfalfaMOGP, MultitaskGaussianLikelihood

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)
benchmark = CurrinExp2D()
# 10 experiments at fidelity 0, 50 at fidelity 1
fidelities = [10, 50]
x, i, y = benchmark.get_init_data(fidelities, rnd_seed=42)

likelihood = MultitaskGaussianLikelihood(num_tasks=2)
booster = fit_lgbm_forest(x, y)
forest = lgbm_to_alfalfa_forest(booster)
space = benchmark.get_space()
forest.initialise(space)
model = AlfalfaMOGP(
    (torch.from_numpy(x), torch.from_numpy(i)),
    torch.from_numpy(y),
    likelihood,
    forest,
    num_tasks=2,
)

fit_gp_adam(model)

model.eval()
likelihood.eval()

x = torch.tensor([[0.5, 0.5]])
propose_fidelity_information_based(model, x, costs=[1, 10])
