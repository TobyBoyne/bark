"""This is a suggestion for speeding up matrix inverses (for MLL)."""

from dataclasses import replace
from typing import Self

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Float, Int

# from bark.likelihood.marginal_log_likelihood import BARKLikelihood
from bark import forest, types
from bark.types import BARKModel


def get_K_inv_logdet(
    bark_model: BARKModel, data: types.Data
) -> tuple[Float[Array, "N N"], Float[Array, ""]]:
    N = data.train_Y.shape[-2]
    K_XX = forest.forest_gram_matrix(
        data.train_X,
        bark_model.trees,
        data.feat_types,
    )
    K_XX_s = K_XX + (1e-6 + bark_model.noise) * jnp.eye(N)
    cholesky = jax.scipy.linalg.cho_factor(K_XX_s)
    K_inv = jax.scipy.linalg.cho_solve(cholesky, jnp.eye(N))
    _, K_logdet = jnp.linalg.slogdet(K_XX)
    return K_inv, K_logdet


def low_rank_inverse_update(
    K_inv: Float[Array, "N N"], U: Float[Array, "N B"], subtract: Bool[Array, ""]
) -> Float[Array, "N N"]:
    mul = jnp.where(subtract, -1.0, 1.0)
    den = mul * jnp.eye(U.shape[-1]) + (U.T @ K_inv @ U)
    return K_inv - K_inv @ U @ jnp.linalg.solve(den, U.T @ K_inv)


def low_rank_det_update(
    K_inv: Float[Array, "N N"],
    U: Float[Array, "N B"],
    K_logdet: Float[Array, ""],
    subtract: Bool[Array, ""],
) -> Float[Array, ""]:
    mul = jnp.where(subtract, -1.0, 1.0)
    _, logabsdet = jnp.linalg.slogdet(jnp.eye(U.shape[-1]) + mul * (U.T @ K_inv @ U))
    return K_logdet + logabsdet


def mll_from_inv_and_logdet(
    K_inv: Float[Array, "N N"], K_logdet: Float[Array, ""], data: types.Data
) -> Float[Array, ""]:
    y = data.train_Y
    return 0.5 * (-y.T @ K_inv @ y - K_logdet).squeeze((-2, -1))


def get_similarity_matrix_delta_low_rank(
    trees: types.Tree, new_trees: types.Tree, data: types.Data
) -> Bool[Array, "N B m"]:
    """Get the low rank representation of the change in similarity matrix.

    The change in similarity matrix due to sampling new trees can be written as U @ U.T,
    where U has shape (N, B) and B is the number of unique leaves."""

    # TODO: try with jnp.unique as well, since it's very unlikely that all 128 leaves
    # are being used
    train_X = data.train_X
    m = trees.feature_idx.shape[-2]

    leaves = forest.pass_through_forest(
        train_X,
        trees,
        data.feat_types,
    )  # N m
    new_leaves = forest.pass_through_forest(
        train_X,
        new_trees,
        data.feat_types,
    )
    leaf_idcs = jnp.arange(trees.feature_idx.shape[-1])[None, :, None]  # 1 B 1
    U = jnp.equal(leaves[:, None, :], leaf_idcs).astype(jnp.float64)  # N B m
    U_new = jnp.equal(new_leaves[:, None, :], leaf_idcs).astype(jnp.float64)
    return (U_new - U) / m


# TODO: create a shared BARKLikelihood class
@struct.dataclass
class WoodburyBARKLikelihood:
    """Uses the Woodbury identity to quickly perform a low-rank update.

    Computes a low-rank update to the matrix inverse, using
    (K + U UT)^-1 = K^-1 (I - U(UT K^-1 U + I)^-1 UT K^-1)
    and
    log|K + U UT| = log|K| + log|I + UT K^-1 U|"""

    mll: Float[Array, ""]
    similarity_matrix_delta: Float[Array, "N B m"]
    K_inv: Float[Array, "N N"]
    K_logdet: Float[Array, ""]

    @classmethod
    def create_from_bark_model(cls, bark_model: BARKModel, data: types.Data) -> Self:
        K_inv, K_logdet = get_K_inv_logdet(bark_model, data)
        mll = mll_from_inv_and_logdet(K_inv, K_logdet, data)
        U_shape = (
            data.train_Y.shape[-2],
            bark_model.trees.feature_idx.shape[-1],
            bark_model.num_trees,
        )
        similarity_matrix_delta = jnp.zeros(U_shape)

        return cls(
            mll=mll,
            similarity_matrix_delta=similarity_matrix_delta,
            K_inv=K_inv,
            K_logdet=K_logdet,
        )

    def compute_new_tree_likelihood(
        self, bark_model: BARKModel, tree_idx: Int[Array, ""], data: types.Data
    ) -> Self:
        subtract = jnp.bool(False)
        U = self.similarity_matrix_delta[..., tree_idx]
        inv_update = low_rank_inverse_update(self.K_inv, U, subtract)
        logdet_update = low_rank_det_update(self.K_inv, U, self.K_logdet, subtract)
        mll_update = mll_from_inv_and_logdet(inv_update, logdet_update, data)
        return replace(self, mll=mll_update, K_inv=inv_update, K_logdet=logdet_update)

    def compute_new_noise_likelihood(
        self, bark_model: BARKModel, data: types.Data
    ) -> Self:
        # could also cache current K for quicker computation of MLL here
        K_inv, K_logdet = get_K_inv_logdet(bark_model, data)
        mll = mll_from_inv_and_logdet(K_inv, K_logdet, data)
        return replace(self, mll=mll, K_inv=K_inv, K_logdet=K_logdet)

    def compute_similarity_matrix_delta(
        self, trees: types.Tree, new_trees: types.Tree, data: types.Data
    ) -> Self:
        U = get_similarity_matrix_delta_low_rank(trees, new_trees, data)
        return replace(self, similarity_matrix_delta=U)

    def update_from_likelihood(self, other_likelihood: Self, accept: Bool[Array, ""]):
        return type(self)(
            mll=jnp.where(accept, other_likelihood.mll, self.mll),
            K_inv=jnp.where(accept, other_likelihood.K_inv, self.K_inv),
            K_logdet=jnp.where(accept, other_likelihood.K_logdet, self.K_logdet),
            # similarity matrix delta will always be the same between two steps
            similarity_matrix_delta=self.similarity_matrix_delta,
        )
