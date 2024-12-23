import gpytorch as gpy
import numpy as np
import torch

from bark.forest import forest_gram_matrix


class TreeAgreementKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, forest: np.ndarray, feat_types: np.ndarray):
        super().__init__()
        self.forest = forest
        self.feat_types = feat_types

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return torch.as_tensor(
            forest_gram_matrix(
                self.forest, x1.detach().numpy(), x2.detach().numpy(), self.feat_types
            )
        )
