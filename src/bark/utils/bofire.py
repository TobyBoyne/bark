import jax.numpy as jnp
from bofire.data_models.domain.api import Inputs
from jaxtyping import ArrayLike, Float

from bark import types
from bofire_mixed.domain import get_feature_bounds, get_feature_types_array


def create_data_from_bofire_inputs(
    train_X: Float[ArrayLike, "N d"], train_Y: Float[ArrayLike, "N 1"], inputs: Inputs
):
    bounds = [get_feature_bounds(feat, encoding="bitmask") for feat in inputs.get()]
    bounds = jnp.asarray(bounds)

    feat_types = jnp.asarray(get_feature_types_array(inputs))

    return types.Data(
        train_X=jnp.asarray(train_X),
        train_Y=jnp.asarray(train_Y),
        bounds=bounds,
        feat_types=feat_types,
    )
