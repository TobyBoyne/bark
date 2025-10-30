from abc import ABC, abstractmethod
from typing import Self

from flax import struct
from jaxtyping import Array, Bool, Float, Int

from bark import types
from bark.types import BARKModel


@struct.dataclass
class BARKLikelihood(ABC):
    mll: Float[Array, ""]

    @classmethod
    @abstractmethod
    def create_from_bark_model(
        cls, bark_model: BARKModel, data: types.Data
    ) -> Self: ...

    @abstractmethod
    def compute_new_tree_likelihood(
        self, bark_model: BARKModel, tree_idx: Int[Array, ""], data: types.Data
    ) -> Self: ...

    @abstractmethod
    def compute_new_noise_likelihood(
        self, bark_model: BARKModel, data: types.Data
    ) -> Self: ...

    @abstractmethod
    def compute_cache_from_new_tree_proposals(
        self, trees: types.Tree, new_trees: types.Tree, data: types.Data
    ) -> Self: ...

    @abstractmethod
    def update_from_likelihood(
        self, other_likelihood, accept: Bool[Array, ""]
    ) -> Self: ...
