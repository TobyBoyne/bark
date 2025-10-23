from dataclasses import replace

import jax
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
class Trees:
    feature_idx: Int[Array, "m max_nodes"]
    threshold: Float[Array, "m max_nodes"]


@struct.dataclass
class BARKModel:
    forest: Trees
    noise: Float[Array, " *batch"]

    def get_flat_model(self):
        forest_reshape = (-1, *self.forest.feature_idx.shape[-2:])
        flat_feature_idx = self.forest.feature_idx.reshape(*forest_reshape)
        flat_threshold = self.forest.threshold.reshape(*forest_reshape)
        return replace(
            self,
            noise=self.noise.reshape(-1),
            forest=replace(
                self.forest,
                feature_idx=flat_feature_idx,
                threshold=flat_threshold,
            ),
        )

    @property
    def num_trees(self):
        return self.forest.threshold.shape[-2]


@struct.dataclass
class BARKConfig:
    warmup_steps: int
    num_samples: int
    steps_per_sample: int
    num_chains: int
    alpha: float
    beta: float
    proposal_weights: Float[jax.Array, " 3"]
    verbose: bool
    use_softplus_transform: bool
    gamma_prior_shape: float
    gamma_prior_rate: float
