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
def _pass_one_through_tree(nodes, X, feat_types):
    node_idx = 0
    while True:
        node = nodes[node_idx]
        feature_idx = node["feature_idx"]

        if node["is_leaf"]:
            return node_idx

        if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
            cond = X[feature_idx] == node["threshold"]
        else:
            cond = X[feature_idx] <= node["threshold"]

        if cond:
            node_idx = node["left"]
        else:
            node_idx = node["right"]


@njit(parallel=True)
def pass_through_tree(nodes, X, feat_types):
    out = np.empty(X.shape[0], dtype=np.uint32)
    for i in prange(X.shape[0]):
        out[i] = _pass_one_through_tree(nodes, X[i], feat_types)
    return out


@njit(parallel=True)
def pass_through_forest(
    nodes,
    X,
    feat_types,
):
    out = np.empty((X.shape[0], nodes.shape[0]), dtype=np.uint32)
    for i in prange(nodes.shape[0]):
        out[:, i] = pass_through_tree(nodes[i], X, feat_types)
    return out


@njit
def get_leaf_vectors(nodes, X, feat_types):
    x_leaves = pass_through_tree(nodes, X, feat_types)
    all_leaves = np.unique(x_leaves)
    leaf_vector = (np.equal(x_leaves[:, None], all_leaves[None, :])).astype(np.float64)
    return leaf_vector


@njit
def forest_gram_matrix(
    nodes,
    x1,
    x2,
    feat_types,
):
    x1_leaves = pass_through_forest(nodes, x1, feat_types)
    x2_leaves = pass_through_forest(nodes, x2, feat_types)
    sim_mat = np.equal(x1_leaves[:, None, :], x2_leaves[None, :, :])  # N x M x m
    sim_mat = 1 / nodes.shape[0] * np.sum(sim_mat, axis=-1)
    return sim_mat


@njit(parallel=True)
def batched_forest_gram_matrix(nodes, x1, x2, feat_types):
    batch_dim = nodes.shape[-3]
    sim_mat = np.zeros((batch_dim, x1.shape[0], x2.shape[0]), dtype=np.float64)
    for i in prange(batch_dim):
        sim_mat[i] = forest_gram_matrix(nodes[i], x1, x2, feat_types)
    return sim_mat


def create_empty_forest(m: int, node_limit: int = 100):
    forest = np.zeros((m, node_limit), dtype=NODE_RECORD_DTYPE)
    forest[:, 0] = (1, 0, 0, 0, 0, 0, 1)
    return forest
