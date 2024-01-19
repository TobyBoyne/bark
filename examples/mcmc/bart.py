from alfalfa import AlfalfaTree, AlfalfaForest
from alfalfa.tree_models.tree_kernels import ATGP, AFGP
from alfalfa.tree_models.forest import Node
from alfalfa.fitting.bart.bart import BART
from alfalfa.fitting.bart.data import Data
from alfalfa.leaf_gp.space import Space, Dimension
from alfalfa.fitting.bart.params import BARTTrainParams

import math
import torch
import gpytorch
from matplotlib import pyplot as plt


# Training data is 11 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 4)
train_x = torch.tensor([0.0, 0.1, 0.3, 0.9]).reshape(-1, 1)
# True function is sin(2*pi*x) with Gaussian noise
torch.manual_seed(42)
train_y = (torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2).flatten()

tree = AlfalfaTree(depth=2)
tree.initialise_tree([0])
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
model = ATGP(train_x, train_y, likelihood, tree)

space = Space([[0.0, 1.0]])
data = Data(space, train_x)
params = BARTTrainParams()
bart = BART(model, data, params)
bart.run()
