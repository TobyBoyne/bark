"""Utils for working with Bofire domains"""

from typing import Literal

import numpy as np
from bofire.data_models.domain.api import Domain, Features, Inputs, Outputs
from bofire.data_models.features.api import (
    AnyFeature,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)

from bark.forest import FeatureTypeEnum


def get_feature_by_index(
    features: Features | Inputs | Outputs, index: int
) -> AnyFeature:
    return features.get().features[index]


def get_index_by_feature_key(features: Features, key: str) -> int:
    return features.get().features.index(features.get_by_key(key))


def get_feature_bounds(
    feature: AnyFeature, encoding: Literal["bitmask", "ordinal"] | None = None
) -> tuple[float, float] | list[str] | list[int]:
    if isinstance(feature, CategoricalInput):
        cats = feature.categories
        bitmask_ub = (1 << len(cats)) - 1
        if encoding == "bitmask":
            return (0, bitmask_ub)
        elif encoding == "ordinal":
            return list(range(len(cats)))
        return cats
    elif isinstance(feature, DiscreteInput):
        return (feature.lower_bound, feature.upper_bound)
    elif isinstance(feature, ContinuousInput):
        return feature.bounds

    raise TypeError(f"Cannot get bounds for feature of type {feature.type}")


def get_cat_idx_from_domain(domain: Domain) -> set[int]:
    """Get the indices of categorical features"""
    return {
        i
        for i, feat in enumerate(domain.inputs.get())
        if isinstance(feat, CategoricalInput)
    }


def get_feature_types_array(domain: Domain) -> np.ndarray:
    return np.array(
        [
            FeatureTypeEnum.Cat.value
            if isinstance(feat, CategoricalInput)
            else FeatureTypeEnum.Int.value
            if isinstance(feat, DiscreteInput)
            else FeatureTypeEnum.Cont.value
            for feat in domain.inputs.get()
        ]
    )


def build_integer_input(*, key: str, unit: str | None = None, bounds: tuple[int, int]):
    lb, ub = bounds
    values = list(range(lb, ub + 1))
    return DiscreteInput(key=key, unit=unit, values=values)
