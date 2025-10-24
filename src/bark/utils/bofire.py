import jax.numpy as jnp
from bofire.data_models.domain.api import Domain

from bark import types
from bofire_mixed.domain import get_feature_bounds, get_feature_types_array


def create_data_from_domain(train_X, train_Y, domain: Domain):
    bounds = [
        get_feature_bounds(feat, encoding="bitmask") for feat in domain.inputs.get()
    ]
    bounds = jnp.asarray(bounds)

    feat_types = jnp.asarray(get_feature_types_array(domain))

    return types.Data(
        train_X=train_X,
        train_Y=train_Y,
        bounds=bounds,
        feat_types=feat_types,
    )
