import jax
from flax import struct
from jaxtyping import Array, Float, UInt

FeatTypesT = UInt[Array, " d"]
IndexT = UInt[Array, "..."] | int

BoundsT = Float[jax.Array, "2 d"]


@struct.dataclass
class Features:
    bounds: BoundsT
    feat_types: FeatTypesT


@struct.dataclass
class Data:
    train_X: Float[jax.Array, "N d"]
    train_Y: Float[jax.Array, "N 1"]
    # TODO: change this to Features
    bounds: BoundsT
    feat_types: FeatTypesT
