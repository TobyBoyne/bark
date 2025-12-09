from typing import Self

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Float, Int


@struct.dataclass
class Tree:
    feature_idx: Int[Array, "*batch max_nodes"]
    threshold: Float[Array, "*batch max_nodes"]

    def __getitem__(self, idx) -> Self:
        return self.__class__(
            feature_idx=self.feature_idx[idx], threshold=self.threshold[idx]
        )

    def __iter__(self):
        return TreeIterator(self)


@struct.dataclass
class SoftTree(Tree):
    tau: Float[Array, "*batch"]

    def __getitem__(self, idx) -> Self:
        return self.__class__(
            feature_idx=self.feature_idx[idx],
            threshold=self.threshold[idx],
            tau=self.tau,
        )


class TreeIterator:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.tree_index = -1
        shape = tree.feature_idx.shape
        assert len(shape) == 2
        self.m = shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.tree_index + 1 >= self.m:
            raise StopIteration
        self.tree_index += 1
        return self.tree[self.tree_index]


@struct.dataclass
class BARKModel:
    trees: Tree
    noise_var: Float[Array, " *batch"]

    def get_flattened_samples(self):
        return jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[len(self.batch_shape) :]), self
        )

    @property
    def num_trees(self):
        return self.trees.threshold.shape[-2]

    @property
    def batch_shape(self):
        return self.noise_var.shape

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
            noise_var=self.noise_var,
        )

    def update_noise(
        self, other_noise: Float[Array, ""], accept: Bool[Array, ""]
    ) -> "BARKModel":
        return BARKModel(
            trees=self.trees, noise_var=jnp.where(accept, other_noise, self.noise_var)
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
    noise_step_size: float = 1.0

    @property
    def proposal_weights(self) -> Float[jax.Array, " 3"]:
        return jnp.array(
            [self.prune_grow_weight, self.prune_grow_weight, self.change_weight]
        )
