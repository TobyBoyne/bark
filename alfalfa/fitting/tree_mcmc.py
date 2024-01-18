from alfalfa import AlfalfaTree, AlfalfaForest
from alfalfa.tree_models.tree_kernels import ATGP, AFGP, AlfalfaGP
from alfalfa.tree_models.forest import Node
from dataclasses import dataclass
import math
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from matplotlib import pyplot as plt
from typing import Union, Optional

from torch.nn import Module as TModule
from torch.distributions import Bernoulli
from gpytorch.priors import UniformPrior, Prior



@dataclass
class MCMCTrainParams:
    num_samples: int = 100
    warmup_steps: int = 100


# Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.

def mcmc_fit(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        model: AlfalfaGP,
        train_params: Optional[MCMCTrainParams] = None
    ):

    if train_params is None:
        train_params = MCMCTrainParams()

    mcmc_run.run(train_x, train_y)

    model.pyro_load_from_samples(mcmc_run.get_samples())
    return mcmc_run