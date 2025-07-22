from dataclasses import replace

import flax.linen as nn
from jaxtyping import Array, Float, Int, UInt

FeatTypesT = UInt[Array, " d"]
IndexT = UInt[Array, " *n"]

FeatureIndexT = Int[Array, "*batch m 2**max_depth"]  # not unsigned as -1 is leaf
ThresholdT = Float[Array, "*batch m 2**max_depth"]


class Trees(nn.Module):
    feature_idx: FeatureIndexT
    threshold: ThresholdT


class BARKModel(nn.Module):
    forest: Trees
    noise: Float[Array, " *batch"]

    def get_flat_model(self):
        forest_reshape = (-1, *self.forest.feature_idx.shape[-2])
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
