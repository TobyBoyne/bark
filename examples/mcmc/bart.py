from alfalfa import AlfalfaTree, AlfalfaForest
from alfalfa.tree_models.tree_kernels import AlfalfaGP
from alfalfa.tree_models.forest import DecisionNode
from alfalfa.fitting.bart.bart import BART
from alfalfa.fitting.bart.data import Data
from alfalfa.leaf_gp.space import Space, Dimension
from alfalfa.fitting.bart.params import BARTTrainParams

import math
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt


# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 20).reshape(-1, 1)
# train_x = torch.tensor([0.0, 0.1, 0.3, 0.9]).reshape(-1, 1)
space = Space([[0.0, 1.0]])

# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
np.random.seed(42)
train_y = (torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2).flatten()

tree = AlfalfaTree(height=1)
data = Data(space, train_x)
tree.initialise(space, data.get_rule_prior())
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
model = AlfalfaGP(train_x, train_y, likelihood, tree)

params = BARTTrainParams()
bart = BART(model, data, params)
bart.run()
