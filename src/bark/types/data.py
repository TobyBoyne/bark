import jax
from flax import struct
from jaxtyping import Array, Float, UInt

from bark.enums import FeatureTypeEnum
from bark.utils.bit_operations import next_power_of_2_exponent

FeatTypesT = UInt[Array, " d"]
IndexT = UInt[Array, "..."] | int

BoundsT = Float[jax.Array, "2 d"]


@struct.dataclass
class Features:
    bounds: BoundsT
    feat_types: FeatTypesT

    @property
    def cat_idcs(self) -> list[int]:
        return [i for i, f in enumerate(self.feat_types) if f == FeatureTypeEnum.Cat]

    @property
    def ordinal_bounds(self):
        cat_idcs = self.cat_idcs
        cat_bounds = self.bounds[1, cat_idcs]
        ordinal_cat_bound = next_power_of_2_exponent(cat_bounds)
        return self.bounds.at[1, cat_idcs].set(ordinal_cat_bound)


@struct.dataclass
class Data:
    train_X: Float[jax.Array, "N d"]
    train_Y: Float[jax.Array, "N 1"]
    # TODO: change this to Features
    bounds: BoundsT
    feat_types: FeatTypesT
