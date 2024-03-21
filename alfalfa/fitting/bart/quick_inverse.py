"""This is a suggestion for speeding up matrix inverses (for MLL).
However, it doesn't seem to be much faster..."""

import numpy as np
import torch
from beartype.typing import TYPE_CHECKING, Optional
from jaxtyping import Float, Shaped

if TYPE_CHECKING:
    from .tree_transitions import Transition
from ...tree_kernels import AlfalfaGP

InverseType = Float[torch.Tensor, "N N"]
DetType = Float[torch.Tensor, ""]


def low_rank_inv_update(
    K_inv: InverseType, U: Float[torch.Tensor, "N B"]
) -> InverseType:
    # num = (K_inv @ z) @ (z.mT @ K_inv)
    # den = 1 + z.mT @ K_inv @ z
    den = torch.eye(U.shape[-1]) + U.mT @ K_inv @ U
    return K_inv - K_inv @ U @ torch.linalg.solve(den, U.mT @ K_inv)


def low_rank_det_update(
    K_inv: InverseType, U: Float[torch.Tensor, "N B"], K_logdet: DetType
) -> DetType:
    # return torch.log(1 + z.mT @ K_inv @ z).squeeze() + K_logdet
    return K_logdet + torch.logdet(torch.eye(U.shape[-1]) + U.mT @ K_inv @ U)


def mll(
    K_inv: InverseType, K_logdet: DetType, y: Float[torch.Tensor, "N 1"]
) -> Float[torch.Tensor, ""]:
    return (-y.T @ K_inv @ y - K_logdet).squeeze()


class QuickInverter:
    """Handles the fast inversion of forest kernels.

    Using the Sherman–Morrison formula and the Matrix determinant lemma, matrix inversion
    becomes O(N^2 m), where m is the number of trees in the forest. Since m is fixed, for large
    N this becomes O(N^2)"""

    def __init__(self, model: AlfalfaGP):
        covar = model.covar_module(model.train_inputs[0])
        noise_covar = model.likelihood._shaped_noise_covar(
            torch.Size([covar.shape[-1]])
        )
        init_K = covar + noise_covar

        self._cached_inverse: InverseType = torch.cholesky_inverse(
            torch.linalg.cholesky(init_K).to_dense()
        )
        self._cached_logdet: DetType = torch.logdet(init_K)

        self._new_cached_inverse: Optional[InverseType] = None
        self._new_cached_logdet: Optional[DetType] = None

        self._x: Shaped[np.ndarray, "N D"] = model.train_inputs[0].detach().numpy()
        self._y: Float[torch.Tensor, "N 1"] = model.train_targets.reshape(-1, 1)

    def get_mll_current(self) -> Float[torch.Tensor, ""]:
        K_inv, K_logdet = self._cached_inverse, self._cached_logdet
        return mll(K_inv, K_logdet, self._y)

    def get_mll_proposed(self, transition: "Transition") -> Float[torch.Tensor, ""]:
        tree = transition.tree
        cur_leaf_vectors = torch.as_tensor(tree.get_leaf_vectors(self._x))
        with transition:
            new_leaf_vectors = torch.as_tensor(tree.get_leaf_vectors(self._x))

        K_inv = self.cached_inverse
        K_logdet = self.cached_logdet

        K_inv = low_rank_inv_update(K_inv, -cur_leaf_vectors)
        K_logdet = low_rank_det_update(K_inv, -cur_leaf_vectors, K_logdet)

        K_inv = low_rank_inv_update(K_inv, new_leaf_vectors)
        K_logdet = low_rank_det_update(K_inv, new_leaf_vectors, K_logdet)

        self._new_cached_inverse = K_inv
        self._new_cached_logdet = K_logdet

        return mll(K_inv, K_logdet, self._y)

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

    def cache_proposal(self):
        self._cached_inverse = self._new_cached_inverse
        self._cached_logdet = self._new_cached_logdet
