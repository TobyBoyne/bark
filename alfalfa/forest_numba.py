"""Inspired by https://github.com/ogrisel/pygbm"""


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


@njit
def _pass_one_through_tree(nodes, X, feat_is_cat):
    node_idx = 0
    while True:
        node = nodes[node_idx]

        if node["is_leaf"]:
            return node_idx

        if feat_is_cat[node["feature_idx"]]:
            cond = X[node["feature_idx"]] == node["threshold"]
        else:
            cond = X[node["feature_idx"]] <= node["threshold"]

        if cond:
            node_idx = node["left"]
        else:
            node_idx = node["right"]


@njit(parallel=True)
def _pass_through_tree(nodes, X, feat_is_cat):
    out = np.empty(X.shape[0], dtype=np.uint32)
    for i in prange(X.shape[0]):
        out[i] = _pass_one_through_tree(nodes, X[i], feat_is_cat)
    return out


# nodes = np.array(
#     [
#         (0, 0, 0.5, 1, 2, 0),
#         (0, 0, 0.25, 3, 4, 1),
#         (1, 0, 1.0, 0, 0, 1),
#         (1, 0, 1.0, 0, 0, 2),
#         (1, 0, 1.0, 0, 0, 2),
#     ],
#     dtype=NODE_RECORD_DTYPE,
# )

# X = np.linspace(0, 1, 10_000).reshape(-1, 1)
# feat_is_cat = np.array([False])

# _pass_through_tree(nodes, X, feat_is_cat)
# t = timeit.timeit(lambda: _pass_through_tree(nodes, X, feat_is_cat), number=100)
# print(t)

# print(np.ones((5,), dtype=NODE_RECORD_DTYPE))
