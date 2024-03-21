"""This is a suggestion for speeding up matrix inverses (for MLL).
However, it doesn't seem to be much faster..."""

import numpy as np
import torch
from beartype.typing import Optional
from jaxtyping import Float, Shaped

from ..fitting.bart.tree_transitions import Transition
from ..forest import AlfalfaTree

InverseType = Float[torch.Tensor, "N N"]
DetType = Float[torch.Tensor, ""]


def rank_one_inv_update(
    K_inv: InverseType, z: Float[torch.Tensor, "N 1"]
) -> InverseType:
    num = (K_inv @ z) @ (z.mT @ K_inv)
    den = 1 + z.mT @ K_inv @ z
    return K_inv - num / den


def rank_one_det_update(
    K_inv: InverseType, z: Float[torch.Tensor, "N 1"], K_logdet: DetType
) -> DetType:
    return torch.log(1 + z.mT @ K_inv @ z) + K_logdet


class QuickInverter:
    """Handles the fast inversion of forest kernels.

    Using the Sherman–Morrison formula and the Matrix determinant lemma, matrix inversion
    becomes O(N^2 m), where m is the number of trees in the forest. Since m is fixed, for large
    N this becomes O(N^2)"""

    def __init__(self):
        self._cached_inverse: Optional[InverseType] = None
        self._cached_logdet: Optional[DetType] = None

        self._new_cached_inverse: Optional[InverseType] = None
        self._new_cached_logdet: Optional[DetType] = None

    def tree_transition_update(
        self, x: Shaped[np.ndarray, "N D"], tree: AlfalfaTree, transition: Transition
    ) -> tuple[InverseType, DetType]:
        cur_leaf_vectors = tree.get_leaf_vectors(x)
        with transition:
            new_leaf_vectors = tree.get_leaf_vectors(x)

        K_inv = self.cached_inverse
        K_logdet = self.cached_logdet
        for z in cur_leaf_vectors:
            K_inv = rank_one_inv_update(K_inv, -z)
            K_det = rank_one_det_update(K_inv, z, K_logdet)

        for z in new_leaf_vectors:
            K_inv = rank_one_inv_update(K_inv, z)
            K_det = rank_one_det_update(K_inv, z, K_logdet)

        self._new_cached_inverse: Optional[InverseType] = K_inv
        self._new_cached_logdet: Optional[DetType] = K_logdet

        return K_inv, K_det

    @property
    def cached_inverse(self):
        if self._cached_inverse is None:
            raise ValueError
        return self._cached_inverse

    @property
    def cached_logdet(self):
        if self._cached_logdet is None:
            raise ValueError
        return self._cached_logdet

    def cache(self):
        self._cached_inverse = self._new_cached_inverse
        self._cached_logdet = self._new_cached_logdet
