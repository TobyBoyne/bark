from dataclasses import replace
from typing import Self

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Float, Int, UInt

FeatTypesT = UInt[Array, " d"]
IndexT = UInt[Array, "..."] | int

BoundsT = Float[jax.Array, "2 d"]


@struct.dataclass
class Data:
    train_X: Float[jax.Array, "N d"]
    train_Y: Float[jax.Array, "N 1"]
    bounds: BoundsT
    feat_types: FeatTypesT


@struct.dataclass
class Tree:
    feature_idx: Int[Array, "*batch max_nodes"]
    threshold: Float[Array, "*batch max_nodes"]

    def __getitem__(self, idx) -> Self:
        return self.__class__(
            feature_idx=self.feature_idx[idx], threshold=self.threshold[idx]
        )


@struct.dataclass
class BARKModel:
    trees: Tree
    noise: Float[Array, " *batch"]

    def get_flat_model(self):
        forest_reshape = (-1, *self.trees.feature_idx.shape[-2:])
        flat_feature_idx = self.trees.feature_idx.reshape(*forest_reshape)
        flat_threshold = self.trees.threshold.reshape(*forest_reshape)
        return replace(
            self,
            noise=self.noise.reshape(-1),
            forest=replace(
                self.trees,
                feature_idx=flat_feature_idx,
                threshold=flat_threshold,
            ),
        )

    @property
    def num_trees(self):
        return self.trees.threshold.shape[-2]

    @property
    def batch_shape(self):
        return self.noise.shape

    def update_trees(self, other_trees: Tree, accept: Bool[Array, " m"]) -> "BARKModel":
        # this method is a JIT-compatible version of `self if accept else other`
        return BARKModel(
            trees=Tree(
                feature_idx=jnp.where(
                    accept, other_trees.feature_idx, self.trees.feature_idx
                ),
                threshold=jnp.where(
                    accept, other_trees.threshold, self.trees.threshold
                ),
            ),
            noise=self.noise,
        )

    def update_noise(
        self, other_noise: Float[Array, ""], accept: Bool[Array, ""]
    ) -> "BARKModel":
        return BARKModel(
            trees=self.trees, noise=jnp.where(accept, other_noise, self.noise)
        )


@struct.dataclass
class BARKConfig:
    warmup_steps: int = 100
    num_samples: int = 4
    steps_per_sample: int = 100
    num_chains: int = 4
    alpha: float = 0.95
    beta: float = 2.0
    num_trees: int = 50
    prune_grow_weight: float = 0.5
    change_weight: float = 1.0
    verbose: bool = False
    gamma_prior_shape: float = 1.5
    gamma_prior_rate: float = 5.0

    @property
    def proposal_weights(self) -> Float[jax.Array, " 3"]:
        return jnp.array(
            [self.prune_grow_weight, self.prune_grow_weight, self.change_weight]
        )
