"""Utils for working with Bofire domains"""

from bofire.data_models.domain.api import Features
from bofire.data_models.features.api import AnyFeature


def get_feature_by_index(features: Features, index: int) -> AnyFeature:
    return features.get().features[index]


def get_index_by_feature_key(features: Features, key: str) -> int:
    return features.get().features.index(features.get_by_key(key))
