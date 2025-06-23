"""This is a suggestion for speeding up matrix inverses (for MLL).
However, it doesn't seem to be much faster..."""

import numpy as np
from jaxtyping import Float
from numba import njit

InverseType = Float[np.ndarray, "N N"]
DetType = Float[np.ndarray, ""]


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


@njit
def mll(K_inv: InverseType, K_logdet: DetType, y: Float[np.ndarray, "N 1"]) -> float:
    return 0.5 * (-y.T @ K_inv @ y - K_logdet)[0, 0]
