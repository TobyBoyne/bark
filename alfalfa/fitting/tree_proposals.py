from enum import Enum

import numpy as np
from numba import njit

from alfalfa.forest_numba import NODE_RECORD_DTYPE, FeatureTypeEnum

NODE_PROPOSAL_DTYPE = np.dtype(
    [
        ("node_idx", np.uint32),
        ("prev_feature_idx", np.uint32),
        ("prev_threshold", np.float32),
        ("new_feature_idx", np.uint32),
        ("new_threshold", np.float32),
    ]
)


class TreeProposalEnum(Enum):
    Grow = 0
    Prune = 1
    Change = 2


@njit
def _get_two_inactive_nodes(nodes):
    # return the indices of the first two nodes that are inactive
    # jittable
    found = 0
    inactive = [-1, -1]
    for i, node in enumerate(nodes):
        if node["active"] == 0:
            inactive[found] = i
            found += 1
        if found == 2:
            return inactive

    # TODO: handle no inactive nodes
    raise OverflowError("The tree container is not large enough")


@njit
def sample_splitting_rule(
    bounds: list[list[float]], feat_types: np.ndarray, rng: np.random.Generator
):
    feature_idx = rng.integers(0, len(bounds))
    if feat_types[feature_idx] == FeatureTypeEnum.Cat:
        threshold = rng.integers(0, len(bounds[feature_idx]))
    elif feat_types[feature_idx] == FeatureTypeEnum.Int:
        threshold = rng.integers(
            bounds[feature_idx][0], bounds[feature_idx][1], endpoint=True
        )
    else:
        threshold = rng.uniform(bounds[feature_idx][0], bounds[feature_idx][1])

    return feature_idx, threshold


@njit
def splitting_rule_logprob(
    bounds: list[list[float]], feat_types: np.ndarray, feature_idx: int
):
    if feat_types[feature_idx] == FeatureTypeEnum.Cat:
        threshold_prob = -np.log(len(bounds[feature_idx]))
    elif feat_types[feature_idx] == FeatureTypeEnum.Int:
        threshold_prob = -np.log(bounds[feature_idx][1] - bounds[feature_idx][0] + 1)
    else:
        threshold_prob = -np.log((bounds[feature_idx][1] - bounds[feature_idx][0]))

    return -np.log(len(bounds)) + threshold_prob


@njit
def grow(nodes: np.ndarray, node_proposal):
    nodes = nodes.copy()
    left_idx, right_idx = _get_two_inactive_nodes(nodes)
    depth = nodes[node_proposal["node_idx"]]["depth"] + 1
    nodes[left_idx] = (1, 0, 0, 0, 0, depth, 1)
    nodes[right_idx] = (1, 0, 0, 0, 0, depth, 1)
    nodes[node_proposal["node_idx"]] = (
        0,
        node_proposal["new_feature_idx"],
        node_proposal["new_threshold"],
        left_idx,
        right_idx,
        depth,
        1,
    )
    return nodes


@njit
def prune(nodes: np.ndarray, node_proposal):
    nodes = nodes.copy()
    node = nodes[node_proposal["node_idx"]]
    nodes[node["left"]]["active"] = 0
    nodes[node["right"]]["active"] = 0
    nodes[node_proposal["node_idx"]] = (1, 0, 0, 0, 0, node["depth"], 1)
    return nodes


@njit
def change(nodes: np.ndarray, node_proposal):
    nodes = nodes.copy()
    node = nodes[node_proposal["node_idx"]]
    node["feature_idx"] = node_proposal["new_feature_idx"]
    node["threshold"] = node_proposal["new_threshold"]
    return nodes


@njit
def get_tree_proposal(
    nodes: np.ndarray,
    bounds,
    feat_types,
    tree_rng: np.random.Generator,
    params: np.ndarray,
):
    # node_proposal = np.zeros((), dtype=NODE_PROPOSAL_DTYPE)
    # proposal_type = tree_rng.choice(
    #     [TreeProposalEnum.Grow, TreeProposalEnum.Prune, TreeProposalEnum.Change],
    #     p=params["proposal_weights"],
    # )
    pass
    # node_proposal["node_idx"] = tree_rng.integers(0, len(nodes))
    # node_proposal["prev_feature_idx"] = nodes[node_proposal["node_idx"]]["feature_idx"]
    # node_proposal["prev_threshold"] = nodes[node_proposal["node_idx"]]["threshold"]
    # proposal_type = tree_rng.integers(0, 3)
    # if proposal_type == TreeProposalEnum.Grow:
    #     node_proposal["new_feature_idx"], node_proposal["new_threshold"] = sample_splitting_rule(bounds, feat_types, tree_rng)
    # elif proposal_type == TreeProposalEnum.Prune:
    #     pass
    # else:
    #     node_proposal["new_feature_idx"], node_proposal["new_threshold"] = sample_splitting_rule(bounds, feat_types, tree_rng)
    # return node_proposal


# x = np.array((1, 2, 3, 4, 5), dtype=NODE_PROPOSAL_DTYPE)
# print(x)
# bounds = List([[1, 2, 3], [4, 5, 6], [7, 8]])
# print(foo(y))


nodes = np.zeros((5,), dtype=NODE_RECORD_DTYPE)
nodes[2] = (1, 0, 0.5, 0, 0, 0, 0)
print(nodes)
