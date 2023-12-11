import torch
import gpytorch as gpy
import matplotlib.pyplot as plt
from .alternating_forest import AlternatingTree, AlternatingForest, Node
from ..utils.plots import plot_gp_1d


class AlternatingTreeKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree: AlternatingTree):
        super().__init__()
        self.tree = tree

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return self.tree.gram_matrix(x1, x2)


class AlternatingForestKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, forest: AlternatingForest):
        super().__init__()
        self.forest = forest

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return self.forest.gram_matrix(x1, x2)


class AlternatingGP(gpy.models.ExactGP):
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


class ATGP(AlternatingGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.tree_model = None
        self.mean_module = gpy.means.ZeroMean()

        self.tree = AlternatingTree(depth=3)
        tree_kernel = AlternatingTreeKernel(self.tree)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)


class AFGP(AlternatingGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.tree_model = None
        self.mean_module = gpy.means.ZeroMean()

        self.forest = AlternatingForest(depth=3, num_trees=10)
        forest_kernel = AlternatingForestKernel(self.forest)
        self.covar_module = gpy.kernels.ScaleKernel(forest_kernel)
