import numpy as np

from bark.fitting.tree_proposals import sample_splitting_rule
from bark.forest import FeatureTypeEnum


def test_sample_integer_feature():
    feat_types = np.array([FeatureTypeEnum.Int.value])
    bounds = np.array([[0, 10]])
    for i in range(100):
        feature_idx, threshold = sample_splitting_rule(bounds, feat_types)
        assert bounds[feature_idx, 0] <= threshold < bounds[feature_idx, 1]


def test_sample_integer_feature_invalid():
    feat_types = np.array([FeatureTypeEnum.Int.value])
    bounds = np.array([[5, 5]])
    for i in range(100):
        feature_idx, threshold = sample_splitting_rule(bounds, feat_types)
        assert threshold == bounds[feature_idx, 1]
