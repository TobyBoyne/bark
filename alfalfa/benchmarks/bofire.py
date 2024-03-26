import numpy as np
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)

from ..utils.space import BoundType


def bofire_domain_to_bounds(
    domain: Domain,
) -> tuple[list[list[BoundType]], set[int], set[int]]:
    bounds = []
    cat_idx = set()
    int_idx = set()
    for i, feat in enumerate(domain.inputs.get()):
        if isinstance(feat, ContinuousInput):
            bounds.append(list(feat.bounds))
        elif isinstance(feat, CategoricalInput):
            bounds.append(feat.categories)
            cat_idx.add(i)
        elif isinstance(feat, DiscreteInput):
            assert (
                np.all(np.diff(feat.values) == 1) and feat.values[0] % 1 == 0
            ), "Discrete values must be consecutive integers"

            bounds.append(feat.values)
            int_idx.add(i)
        else:
            raise TypeError(f"Feature {type(feat)} not recognised")
    return bounds, cat_idx, int_idx
