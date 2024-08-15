import numpy as np
from numba import njit
from numba.typed import List

TREE_PROPOSAL_DTYPE = np.dtype(
    [("feature_idx", np.uint32), ("threshold", np.float32), ()]
)


@njit
def sample_splitting_rule(
    bounds: list[list[float]], feat_is_cat: np.ndarray, rng: np.random.Generator
):
    feature_idx = rng.integers(0, len(bounds))
    if feat_is_cat[feature_idx]:
        threshold = rng.integers(0, len(bounds[feature_idx]))
    else:
        threshold = rng.uniform(bounds[feature_idx][0], bounds[feature_idx][1])

    return feature_idx, threshold


@njit
def splitting_rule_logprob(
    bounds: list[list[float]], feat_is_cat: np.ndarray, feature_idx: int
):
    if feat_is_cat[feature_idx]:
        threshold_prob = -np.log(len(bounds[feature_idx]))
    else:
        threshold_prob = -np.log((bounds[feature_idx][1] - bounds[feature_idx][0]))

    return -np.log(len(bounds)) + threshold_prob


@njit
def grow(
    nodes: np.ndarray, bounds: list[list[float]], feat_is_cat: np.ndarray, tree_proposal
):
    pass


bounds = List([[1, 2, 3], [4, 5, 6], [7, 8]])
# print(foo(y))
