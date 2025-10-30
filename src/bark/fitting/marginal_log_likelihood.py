from dataclasses import replace
from typing import Self

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Float, Int

from bark import forest, types
from bark.types import BARKModel


def mll_bark_model(bark_model: BARKModel, data: types.Data) -> Float[Array, ""]:
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
    bark_model: BARKModel,
    K_XX: Float[Array, "N N"],
    data: types.Data,
) -> Float[Array, ""]:
    train_Y = data.train_Y

    K_XX_s = K_XX + (1e-6 + bark_model.noise) * jnp.eye(K_XX.shape[0])
    cholesky = jax.scipy.linalg.cho_factor(K_XX_s)
    K_XX_s_y = jax.scipy.linalg.cho_solve(cholesky, train_Y)

    _, K_logdet = jnp.linalg.slogdet(K_XX_s)
    return 0.5 * (-train_Y.T @ K_XX_s_y - K_logdet).squeeze((-2, -1))


def _get_similarity_matrix(
    trees: types.Tree, data: types.Data
) -> Float[Array, "N N m"]:
    train_X = data.train_X
    leaves = forest.pass_through_forest(train_X, trees, data.feat_types)
    return forest.similarity_matrix(leaves, leaves).astype(jnp.float64)


def get_cached_similarity_matrix_and_delta(
    trees: types.Tree, new_trees: types.Tree, data: types.Data
) -> tuple[Float[Array, "N N"], Float[Array, "N N m"]]:
    m = trees.feature_idx.shape[-2]
    G_XX = _get_similarity_matrix(trees, data)
    G_XX_new = _get_similarity_matrix(new_trees, data)
    G_XX_delta = G_XX_new - G_XX
    return jnp.mean(G_XX, axis=-1), G_XX_delta / m


@struct.dataclass
class BARKLikelihood:
    mll: Float[Array, ""]
    similarity_matrix: Float[Array, "N N"]
    similarity_matrix_delta: Float[Array, "N N m"]

    @classmethod
    def create_from_bark_model(cls, bark_model: BARKModel, data: types.Data) -> Self:
        G_XX = _get_similarity_matrix(bark_model.trees, data)
        G_XX_delta = jnp.zeros_like(G_XX)
        G_XX = jnp.mean(G_XX, axis=-1)
        mll = mll_bark_model_cached_gram(bark_model, G_XX, data)
        return cls(
            mll=mll,
            similarity_matrix=G_XX,
            similarity_matrix_delta=G_XX_delta,
        )

    def compute_new_tree_likelihood(
        self, bark_model: BARKModel, tree_idx: Int[Array, ""], data: types.Data
    ) -> Self:
        K_XX = self.similarity_matrix + self.similarity_matrix_delta[..., tree_idx]
        new_mll = mll_bark_model_cached_gram(bark_model, K_XX, data)
        return type(self)(
            mll=new_mll,
            similarity_matrix=K_XX,
            similarity_matrix_delta=self.similarity_matrix_delta,
        )

    def compute_new_noise_likelihood(
        self, bark_model: BARKModel, data: types.Data
    ) -> Self:
        mll = mll_bark_model_cached_gram(bark_model, self.similarity_matrix, data)
        return replace(self, mll=mll)

    def update_from_likelihood(self, other_likelihood: Self, accept: Bool[Array, ""]):
        return type(self)(
            mll=jnp.where(accept, other_likelihood.mll, self.mll),
            similarity_matrix=jnp.where(
                accept, other_likelihood.similarity_matrix, self.similarity_matrix
            ),
            # similarity matrix delta will always be the same between two steps
            similarity_matrix_delta=self.similarity_matrix_delta,
        )

    def compute_similarity_matrix_delta(
        self, trees: types.Tree, new_trees: types.Tree, data: types.Data
    ) -> Self:
        _, G_XX_delta = get_cached_similarity_matrix_and_delta(trees, new_trees, data)
        return replace(self, similarity_matrix_delta=G_XX_delta)

    # def update_mll(self, other_mll: Float[Array, ""], accept: Bool[Array, ""]) -> Self:
    #     return replace(self, cached_mll=jnp.where(accept, other_mll, self.cached_mll))

    # def update_similarity_matrix(self, other_sim_mat: Float[Array, "N N"], accept: Bool[Array, ""]) -> Self:
    #     return replace(self, cached_similarity_matrix=jnp.where(accept, other_sim_mat, self.cached_similarity_matrix))

    # def update_similarity_matrix_delta(self, other_sim_mat_delta: Float[Array, "N N m"], accept: Bool[Array, ""]) -> Self:
    #     return replace(self, cached_similarity_matrix_delta=jnp.where(accept, other_sim_mat_delta, self.cached_similarity_matrix_delta))
