import gpytorch as gpy
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Optional
from jaxtyping import Array, Float

from bark import forest, types

from .tree_model_kernel import TreeAgreementKernel


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


def forest_predict(
    bark_model: types.BARKModel,
    data: types.Data,
    test_X: Float[Array, "M d"],
    diag: bool = True,
) -> tuple[
    Float[Array, "batch M"], Float[Array, "batch M M"] | Float[Array, "batch M"]
]:
    # flatten model
    bark_model = bark_model.get_flattened_samples()
    num_samples = bark_model.noise.shape[0]
    num_test = test_X.shape[0]

    K_XX = jax.vmap(forest.forest_gram_matrix, in_axes=(None, 0, None))(
        data.train_X,
        bark_model.trees,
        data.feat_types,
    )
    K_XX_s = K_XX + (1e-6 + bark_model.noise[:, None, None]) * np.eye(
        data.train_X.shape[0]
    )

    K_xX = jax.vmap(forest.forest_covar_matrix, in_axes=(None, None, 0, None))(
        data.train_X,
        test_X,
        bark_model.trees,
        data.feat_types,
    )

    cholesky = jax.scipy.linalg.cho_factor(K_XX_s)

    mu = K_xX @ jax.scipy.linalg.cho_solve(cholesky, data.train_Y)
    var = 1.0 - K_xX @ jax.scipy.linalg.cho_solve(cholesky, K_xX.transpose((0, 2, 1)))

    mu = mu.reshape(num_samples, num_test)
    if diag:
        var = jnp.diagonal(var, axis1=1, axis2=2)
    return mu, var


def mixture_of_gaussians_as_normal(
    mu: Float[Array, "batch N"],
    var: Float[Array, "batch N"],
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    r"""Find the mean and variance of a mixture of Gaussians.

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
