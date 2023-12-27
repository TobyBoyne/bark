import torch
import gpytorch as gpy
from .forest import AlfalfaTree, AlfalfaForest


class AlfalfaTreeKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree: AlfalfaTree):
        super().__init__()
        self.tree = tree

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return self.tree.gram_matrix(x1, x2)


class AlfalfaForestKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, forest: AlfalfaForest):
        super().__init__()
        self.forest = forest

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return self.forest.gram_matrix(x1, x2)


class AlfalfaGP(gpy.models.ExactGP):
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


class ATGP(AlfalfaGP):
    def __init__(
        self, train_inputs, train_targets, likelihood, tree_model: AlfalfaTree
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()

        tree_kernel = AlfalfaTreeKernel(tree_model)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)

    @property
    def tree(self):
        return self.covar_module.base_kernel.tree


class AFGP(AlfalfaGP):
    def __init__(
        self, train_inputs, train_targets, likelihood, forest_model: AlfalfaForest
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()

        forest_kernel = AlfalfaForestKernel(forest_model)
        self.covar_module = gpy.kernels.ScaleKernel(forest_kernel,
            outputscale_constraint=gpy.constraints.Interval(0.2, 2.0))

    @property
    def forest(self):
        return self.covar_module.base_kernel.forest
