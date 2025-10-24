import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bark import forest, types


def mll_bark_model(bark_model: types.BARKModel, data: types.Data) -> Float[Array, ""]:
    train_X, train_Y = data.train_X, data.train_Y
    K_XX = forest.forest_gram_matrix(
        train_X,
        train_X,
        bark_model.trees,
        data.feat_types,
    )
    K_XX_s = K_XX + (1e-6 + bark_model.noise) * jnp.eye(K_XX.shape[0])
    cholesky = jax.scipy.linalg.cho_factor(K_XX_s)
    K_XX_s_y = jax.scipy.linalg.cho_solve(cholesky, train_Y)

    _, K_logdet = jnp.linalg.slogdet(K_XX_s)
    return 0.5 * (-train_Y.T @ K_XX_s_y - K_logdet)
