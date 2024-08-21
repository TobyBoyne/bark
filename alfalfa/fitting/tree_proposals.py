from enum import Enum

import numpy as np
from numba import njit

from alfalfa.fitting.tree_traversal import singly_internal_nodes, terminal_nodes
from alfalfa.forest_numba import FeatureTypeEnum

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
def sample_splitting_rule(bounds: list[list[float]], feat_types: np.ndarray):
    feature_idx = np.random.randint(0, len(bounds))
    if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
        threshold = np.random.randint(0, len(bounds[feature_idx]))
    elif feat_types[feature_idx] == FeatureTypeEnum.Int.value:
        threshold = np.random.randint(
            bounds[feature_idx][0], bounds[feature_idx][1] + 1
        )
    else:
        threshold = np.random.uniform(bounds[feature_idx][0], bounds[feature_idx][1])

    return feature_idx, threshold


@njit
def splitting_rule_logprob(
    bounds: list[list[float]], feat_types: np.ndarray, feature_idx: int
):
    # this function might not be necessary
    if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
        threshold_prob = -np.log(len(bounds[feature_idx]))
    elif feat_types[feature_idx] == FeatureTypeEnum.Int.value:
        threshold_prob = -np.log(bounds[feature_idx][1] - bounds[feature_idx][0] + 1)
    else:
        threshold_prob = -np.log((bounds[feature_idx][1] - bounds[feature_idx][0]))

    return -np.log(len(bounds)) + threshold_prob


@njit
def tree_q_ratio(nodes: np.ndarray, proposal_type: int, node_proposal: np.ndarray):
    if proposal_type == TreeProposalEnum.Grow:
        w_0 = np.size(terminal_nodes(nodes))
        parent = nodes[node_proposal["node_idx"]]
        parent_singly_internal = (
            parent["left"]["is_leaf"] and parent["right"]["is_leaf"]
        )
        w_1_star = np.size(singly_internal_nodes(nodes)) + 1 - parent_singly_internal
        return np.log(w_0) - np.log(w_1_star)

    elif proposal_type == TreeProposalEnum.Prune:
        w_0_star = np.size(terminal_nodes(nodes)) - 1
        w_1 = np.size(singly_internal_nodes(nodes))
        return np.log(w_1) - np.log(w_0_star)

    else:
        return 0.0


@njit
def tree_prior_ratio(nodes: np.ndarray, proposal_type: int, params: np.ndarray):
    alpha = params["alpha"]
    beta = params["beta"]
    depth = nodes[params["node_idx"]]["depth"]

    if proposal_type == TreeProposalEnum.Change:
        return 0.0

    prior_ratio = (
        np.log(alpha)
        + 2 * np.log(1 - alpha / (2 + depth) ** beta)
        + -np.log((1 + depth) ** beta - alpha)
    )

    if proposal_type == TreeProposalEnum.Grow:
        return prior_ratio
    else:
        return -prior_ratio


@njit
def grow(nodes: np.ndarray, node_proposal):
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
    node = nodes[node_proposal["node_idx"]]
    nodes[node["left"]]["active"] = 0
    nodes[node["right"]]["active"] = 0
    nodes[node_proposal["node_idx"]] = (1, 0, 0, 0, 0, node["depth"], 1)
    return nodes


@njit
def change(nodes: np.ndarray, node_proposal):
    node = nodes[node_proposal["node_idx"]]
    node["feature_idx"] = node_proposal["new_feature_idx"]
    node["threshold"] = node_proposal["new_threshold"]
    return nodes


@njit
def get_tree_proposal(
    nodes: np.ndarray,
    bounds,
    feat_types,
    params: np.ndarray,
) -> tuple[np.ndarray, float]:
    node_proposal = np.zeros((), dtype=NODE_PROPOSAL_DTYPE)
    # numba doesn't support weighted choice
    proposal_idx = np.searchsorted(
        np.cumsum(params["proposal_weights"]), np.random.rand()
    )
    proposal_type = [
        TreeProposalEnum.Grow,
        TreeProposalEnum.Prune,
        TreeProposalEnum.Change,
    ][proposal_idx]

    if proposal_type == TreeProposalEnum.Grow:
        valid_nodes = terminal_nodes(nodes)
    else:
        valid_nodes = singly_internal_nodes(nodes)

    if len(valid_nodes) == 0:
        return nodes, -np.inf

    node_proposal["node_idx"] = "xyz"
    node_proposal["node_idx"] = np.random.choice(valid_nodes)
    node_proposal["prev_feature_idx"] = nodes[node_proposal["node_idx"]]["feature_idx"]
    node_proposal["prev_threshold"] = nodes[node_proposal["node_idx"]]["threshold"]

    if (
        proposal_type == TreeProposalEnum.Grow
        or proposal_type == TreeProposalEnum.Change
    ):
        (
            node_proposal["new_feature_idx"],
            node_proposal["new_threshold"],
        ) = sample_splitting_rule(bounds, feat_types)

    log_q_ratio = tree_q_ratio(nodes, proposal_type, node_proposal)
    log_prior_ratio = tree_prior_ratio(nodes, proposal_type, params)

    new_nodes = nodes.copy()
    if proposal_type == TreeProposalEnum.Grow:
        new_nodes = grow(new_nodes, node_proposal)
    elif proposal_type == TreeProposalEnum.Prune:
        new_nodes = prune(new_nodes, node_proposal)
    else:
        new_nodes = change(new_nodes, node_proposal)

    log_q_prior = log_q_ratio + log_prior_ratio
    return new_nodes, log_q_prior
