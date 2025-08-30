"""This is a suggestion for speeding up matrix inverses (for MLL)."""

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax import jit
from jaxtyping import Float
from numba import njit

InverseType = Float[np.ndarray, "N N"]
DetType = Float[np.ndarray, ""]
MLLType = Float[np.ndarray, ""]


@njit
def low_rank_inv_update(
    K_inv: InverseType, U: Float[np.ndarray, "N B"], subtract: bool = False
) -> InverseType:
    mul = -1.0 if subtract else 1.0
    den = mul * np.eye(U.shape[-1]) + (U.T @ K_inv @ U)

    # This line raises a NumbaPerformanceWarning
    # https://github.com/numba/numba/issues/6998
    return K_inv - K_inv @ U @ np.linalg.solve(den, U.T @ K_inv)


@njit
def low_rank_det_update(
    K_inv: InverseType,
    U: Float[np.ndarray, "N B"],
    K_logdet: DetType,
    subtract: bool = False,
) -> DetType:
    mul = -1.0 if subtract else 1.0
    _, logabsdet = np.linalg.slogdet(np.eye(U.shape[-1]) + mul * (U.T @ K_inv @ U))
    return K_logdet + logabsdet


# @njit
# def mll(K_inv: InverseType, K_logdet: DetType, y: Float[np.ndarray, "N 1"]) -> float:
#     return 0.5 * (-y.T @ K_inv @ y - K_logdet)[0, 0]


@jit
def mll(K_inv: InverseType, K_logdet: DetType, y: Float[np.ndarray, "N 1"]) -> float:
    return 0.5 * (-y.T @ K_inv @ y - K_logdet).item()


class LowRankInverter(nn.Module):
    """Uses the Sherman-Morrison identity to quickly perform a low-rank update.

    Computes a low-rank update to the matrix inverse, using
    (K + U UT)^-1 = K^-1 (I - U(UT K^-1 U + I)^-1 UT K^-1)
    and
    log|K + U UT| = log|K| + log|I + UT K^-1 U|"""

    K_inv: InverseType
    K_logdet: DetType
    mll: MLLType
    U: Float[np.ndarray, "N B"]
    subtract: bool

    y: Float[np.ndarray, "N 1"]

    def _low_rank_inverse_update(self) -> InverseType:
        mul = -1.0 if self.subtract else 1.0
        den = mul * jnp.eye(self.U.shape[-1]) + (self.U.T @ self.K_inv @ self.U)

        return -self.K_inv @ self.U @ np.linalg.solve(den, self.U.T @ self.K_inv)

    def _low_rank_logdet_update(self) -> DetType:
        mul = -1.0 if self.subtract else 1.0
        _, logabsdet = jnp.linalg.slogdet(
            jnp.eye(self.U.shape[-1]) + mul * (self.U.T @ self.K_inv @ self.U)
        )
        return logabsdet

    def _low_rank_mll_update(
        self, inv_update: InverseType, logdet_update: DetType
    ) -> MLLType:
        return 0.5 * (-self.y.T @ inv_update @ self.y - logdet_update)

    def low_rank_update(self):
        inv_update = self._low_rank_inverse_update()
        logdet_update = self._low_rank_logdet_update()
        mll_update = self._low_rank_mll_update(inv_update, logdet_update)
        return LowRankInverter(
            K_inv=self.K_inv + inv_update,
            K_logdet=self.K_logdet + logdet_update,
            mll=self.mll + mll_update,
            U=self.U,
            subtract=not self.subtract,
            y=self.y,
        )

    def set_low_rank_update_matrix(self, U: Float[np.ndarray, "N B"], subtract: bool):
        return LowRankInverter(
            K_inv=self.K_inv,
            K_logdet=self.K_logdet,
            mll=self.mll,
            U=U,
            subtract=subtract,
            y=self.y,
        )
