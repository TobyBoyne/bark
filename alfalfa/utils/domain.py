"""Utils for working with Bofire domains"""

from bofire.data_models.domain.api import Domain, Features
from bofire.data_models.features.api import (
    AnyFeature,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)


def get_feature_by_index(features: Features, index: int) -> AnyFeature:
    return features.get().features[index]


def get_index_by_feature_key(features: Features, key: str) -> int:
    return features.get().features.index(features.get_by_key(key))


def get_feature_bounds(feature: AnyFeature) -> tuple[float, float] | list[str]:
    if isinstance(feature, CategoricalInput):
        return feature.categories
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
