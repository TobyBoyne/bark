"""Inspired by https://github.com/ogrisel/pygbm"""


from enum import Enum

import numpy as np
from numba import njit, prange

NODE_RECORD_DTYPE = np.dtype(
    [
        ("is_leaf", np.uint8),
        ("feature_idx", np.uint32),
        ("threshold", np.float32),
        ("left", np.uint32),
        ("right", np.uint32),
        ("depth", np.uint32),
        ("active", np.uint8),
    ]
)


class FeatureTypeEnum(Enum):
    Cat = 0
    Int = 1
    Cont = 2


@njit
def _pass_one_through_tree(nodes, X, feat_types: np.ndarray):
    node_idx = 0
    while True:
        node = nodes[node_idx]
        feature_idx = node["feature_idx"]

        if node["is_leaf"]:
            return node_idx

        if feat_types[feature_idx] == FeatureTypeEnum.Cat:
            cond = X[feature_idx] == node["threshold"]
        else:
            cond = X[feature_idx] <= node["threshold"]

        if cond:
            node_idx = node["left"]
        else:
            node_idx = node["right"]


@njit(parallel=True)
def pass_through_tree(nodes, X, feat_types: np.ndarray):
    out = np.empty(X.shape[0], dtype=np.uint32)
    for i in prange(X.shape[0]):
        out[i] = _pass_one_through_tree(nodes, X[i], feat_types)
    return out


@njit
def get_leaf_vectors(nodes, X, feat_types):
    x_leaves = pass_through_tree(nodes, X, feat_types)
    all_leaves = np.unique(x_leaves)
    leaf_vector = (np.equal(x_leaves[:, None], all_leaves[None, :])).astype(float)
    return leaf_vector


nodes = np.array(
    [
        (0, 0, 0.5, 1, 2, 0, 1),
        (0, 0, 0.25, 3, 4, 1, 1),
        (1, 0, 1.0, 0, 0, 1, 1),
        (1, 0, 1.0, 0, 0, 2, 1),
        (1, 0, 1.0, 0, 0, 2, 1),
    ],
    dtype=NODE_RECORD_DTYPE,
)

X = np.linspace(0, 1, 10_000).reshape(-1, 1)
feat_types = np.array([FeatureTypeEnum.Cont.value])

# t = timeit.timeit(lambda: _pass_through_tree(nodes, X, feat_is_cat), number=100)
# print(t)

# print(np.ones((5,), dtype=NODE_RECORD_DTYPE))
