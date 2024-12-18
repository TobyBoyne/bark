from typing import NamedTuple

import gpytorch as gpy
import numpy as np
from beartype.typing import Optional
from bofire.data_models.domain.api import Domain

from bark.bofire_utils.domain import get_feature_types_array
from bark.forest import FeatureTypeEnum, batched_forest_gram_matrix

from .tree_model_kernel import TreeAgreementKernel


class BARKModel(NamedTuple):
    forest: np.ndarray
    noise: np.ndarray
    scale: np.ndarray


class LeafGP(gpy.models.ExactGP):
    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        forest: np.ndarray,
        feat_types: Optional[np.ndarray] = None,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()

        if feat_types is None:
            feat_types = np.full((train_inputs.shape[0],), FeatureTypeEnum.Cont.value)
        tree_kernel = TreeAgreementKernel(forest, feat_types)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def forest(self) -> np.ndarray:
        return self.covar_module.base_kernel.forest


class LeafMOGP(gpy.models.ExactGP):
    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        forest: np.ndarray,
        feat_types: Optional[np.ndarray] = None,
        num_tasks: int = 2,
    ):
        super().__init__(train_inputs, train_targets, likelihood)

        if feat_types is None:
            feat_types = np.full((train_inputs.shape[0],), FeatureTypeEnum.Cont.value)

        self.mean_module = gpy.means.ZeroMean()
        self.covar_module = TreeAgreementKernel(forest, feat_types)
        self.task_covar_module = gpy.kernels.IndexKernel(num_tasks=num_tasks, rank=1)
        self.num_tasks = num_tasks

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)

        covar = covar_x.mul(covar_i)
        return gpy.distributions.MultivariateNormal(mean_x, covar)

    @property
    def tree_model(self) -> np.ndarray:
        return self.covar_module.tree_model


def forest_predict(
    model: BARKModel,
    data: tuple[np.ndarray, np.ndarray],
    candidates: np.ndarray,
    domain: Domain,
    diag: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    forest, noise, scale = model
    forest = forest.reshape(-1, *forest.shape[-2:])
    noise = noise.reshape(-1)
    scale = scale.reshape(-1)

    num_samples = scale.shape[0]
    num_candidates = candidates.shape[0]

    train_x, train_y = data
    feature_types = get_feature_types_array(domain)
    K_XX = scale[:, None, None] * batched_forest_gram_matrix(
        forest, train_x, train_x, feature_types
    )
    K_XX_s = K_XX + noise[:, None, None] * np.eye(train_x.shape[0])

    K_inv = np.linalg.inv(K_XX_s)
    K_xX = scale[:, None, None] * batched_forest_gram_matrix(
        forest, candidates, train_x, feature_types
    )

    mu = K_xX @ K_inv @ train_y
    var = scale[:, None, None] - K_xX @ K_inv @ K_xX.transpose((0, 2, 1))

    mu = mu.reshape(num_samples, num_candidates)
    if diag:
        var = np.diagonal(var, axis1=1, axis2=2)
    return mu, var


def mixture_of_gaussians_as_normal(
    mu: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the mean and variance of a mixture of Gaussians.

    Since we take samples from the posterior, where each sample has a normal
    distribution over the output, we obtain a prediction that is a mixture of
    Gaussians. If f(y) = \sum_j (1/J)*N(y; mu_j, var_j), then the mean and variance
    of f(y) are given by:
    E[Y] = (1/J) * \sum_j mu_j
    Var[Y] = (1/J) * \sum_j {var_j + \mu_j^2} - ((1/J) * \sum_j mu_j)^2
    """
    mu_y = np.mean(mu, axis=0)
    var_y = np.mean(var + mu**2, axis=0) - mu_y**2
    return mu_y, var_y
