from dataclasses import replace

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Int, UInt

FeatTypesT = UInt[Array, " d"]
IndexT = UInt[Array, "..."]

DataT = tuple[Float[jax.Array, "N d"], Float[jax.Array, "N 1"]]
BoundsT = Float[jax.Array, "2 d"]


@struct.dataclass
class Tree:
    feature_idx: Int[Array, " max_nodes"]
    threshold: Float[Array, " max_nodes"]


@struct.dataclass
class Trees(Tree):
    feature_idx: Int[Array, "m max_nodes"]
    threshold: Float[Array, "m max_nodes"]


@struct.dataclass
class BARKModel:
    trees: Trees
    noise: Float[Array, ""]

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


@struct.dataclass
class BARKConfig:
    warmup_steps: int = 100
    num_samples: int = 4
    steps_per_sample: int = 100
    num_chains: int = 4
    alpha: float = 0.95
    beta: float = 20.0
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
