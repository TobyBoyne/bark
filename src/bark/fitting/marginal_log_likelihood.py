import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bark import forest, types


def mll_bark_model(bark_model: types.BARKModel, data: types.Data) -> Float[Array, ""]:
    train_X, train_Y = data.train_X, data.train_Y
    K_XX = forest.forest_gram_matrix(
        train_X,
        bark_model.trees,
        data.feat_types,
    )
    K_XX_s = K_XX + (1e-6 + bark_model.noise) * jnp.eye(K_XX.shape[0])
    cholesky = jax.scipy.linalg.cho_factor(K_XX_s)
    K_XX_s_y = jax.scipy.linalg.cho_solve(cholesky, train_Y)

    _, K_logdet = jnp.linalg.slogdet(K_XX_s)
    return 0.5 * (-train_Y.T @ K_XX_s_y - K_logdet).squeeze((-2, -1))


def mll_bark_model_cached_gram(
    bark_model: types.BARKModel,
    G_XX: Float[Array, "N N"],
    G_XX_delta: Float[Array, "N N"],
    data: types.Data,
) -> Float[Array, ""]:
    train_Y = data.train_Y
    K_XX = G_XX + G_XX_delta

    K_XX_s = K_XX + (1e-6 + bark_model.noise) * jnp.eye(K_XX.shape[0])
    cholesky = jax.scipy.linalg.cho_factor(K_XX_s)
    K_XX_s_y = jax.scipy.linalg.cho_solve(cholesky, train_Y)

    _, K_logdet = jnp.linalg.slogdet(K_XX_s)
    return 0.5 * (-train_Y.T @ K_XX_s_y - K_logdet).squeeze((-2, -1))


def get_cached_grams(
    trees: types.Tree, new_trees: types.Tree, data: types.Data
) -> tuple[Float[Array, "N N"], Float[Array, "N N m"]]:
    train_X = data.train_X
    m = trees.feature_idx.shape[-2]
    leaves = forest.pass_through_forest(train_X, trees, data.feat_types)
    leaves_new = forest.pass_through_forest(train_X, new_trees, data.feat_types)
    G_XX = forest.similarity_matrix(leaves, leaves).astype(jnp.float32)
    G_XX_new = forest.similarity_matrix(leaves_new, leaves_new).astype(jnp.float32)
    G_XX_delta = G_XX_new - G_XX
    return jnp.mean(G_XX, axis=-1), G_XX_delta / m
