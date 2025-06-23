from enum import Enum
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass

from bark.fitting.tree_traversal import (
    get_node_subspace,
    singly_internal_nodes,
    terminal_nodes,
)
from bark.forest import FeatureTypeEnum
from bark.utils.bit_operations import sample_binary_mask

if TYPE_CHECKING:
    from bark.fitting.bark_sampler import BARKTrainParamsNumba


@jitclass(
    [
        ("node_idx", nb.uint32),
        ("prev_feature_idx", nb.uint32),
        ("prev_threshold", nb.float32),
        ("new_feature_idx", nb.uint32),
        ("new_threshold", nb.float32),
    ]
)
class NodeProposal:
    def __init__(self):
        self.node_idx = 0
        self.prev_feature_idx = 0
        self.prev_threshold = 0
        self.new_feature_idx = 0
        self.new_threshold = 0


class TreeProposalEnum(Enum):
    Grow = 0
    Prune = 1
    Change = 2


@njit
def _get_two_inactive_nodes(nodes):
    # return the indices of the first two nodes that are inactive
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
def _assign_node(
    target, is_leaf, feature_idx, threshold, left, right, parent, depth, active
) -> None:
    # numba requires individual assignment
    # https://numba.discourse.group/t/assigning-to-numpy-structural-array-using-a-tuple-in-jitclass/549/6

    target["is_leaf"] = is_leaf
    target["feature_idx"] = feature_idx
    target["threshold"] = threshold
    target["left"] = left
    target["right"] = right
    target["parent"] = parent
    target["depth"] = depth
    target["active"] = active


@njit
def sample_splitting_rule(bounds: np.ndarray, feat_types: np.ndarray):
    feature_idx = np.random.randint(0, bounds.shape[0])
    if feat_types[feature_idx] == FeatureTypeEnum.Cat.value:
        mask = int(bounds[feature_idx, 1])
        threshold = sample_binary_mask(mask)

    elif feat_types[feature_idx] == FeatureTypeEnum.Int.value:
        if bounds[feature_idx, 0] == bounds[feature_idx, 1]:
            # when the bounds are equal, there is no meaningful split
            # use the upper bound as an indicator
            threshold = bounds[feature_idx, 1]
        else:
            threshold = np.random.randint(
                int(bounds[feature_idx, 0]), int(bounds[feature_idx, 1])
            )
    else:
        threshold = np.random.uniform(bounds[feature_idx, 0], bounds[feature_idx, 1])

    return feature_idx, threshold


@njit
def tree_q_ratio(
    nodes: np.ndarray, proposal_type: TreeProposalEnum, node_proposal: NodeProposal
):
    if proposal_type == TreeProposalEnum.Grow:
        w_0 = np.shape(terminal_nodes(nodes))[0]
        new_nodes = grow(nodes.copy(), node_proposal)
        w_1_star = np.shape(singly_internal_nodes(new_nodes))[0]

        return np.log(w_0) - np.log(w_1_star)

    elif proposal_type == TreeProposalEnum.Prune:
        w_0_star = np.shape(terminal_nodes(nodes))[0] - 1
        w_1 = np.shape(singly_internal_nodes(nodes))[0]
        return np.log(w_1) - np.log(w_0_star)

    else:
        return 0.0


@njit
def tree_prior_ratio(
    nodes: np.ndarray,
    proposal_type: int,
    node_proposal: NodeProposal,
    params: "BARKTrainParamsNumba",
):
    alpha = params.alpha
    beta = params.beta
    depth = nodes[node_proposal.node_idx]["depth"]

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
def grow(nodes: np.ndarray, node_proposal: NodeProposal):
    left_idx, right_idx = _get_two_inactive_nodes(nodes)
    depth = nodes[node_proposal.node_idx]["depth"]
    child_node = (1, 0, 0, 0, 0, node_proposal.node_idx, depth + 1, 1)
    new_parent_node = (
        0,
        node_proposal.new_feature_idx,
        node_proposal.new_threshold,
        left_idx,
        right_idx,
        nodes[node_proposal.node_idx]["parent"],
        depth,
        1,
    )

    _assign_node(nodes[left_idx], *child_node)
    _assign_node(nodes[right_idx], *child_node)
    _assign_node(nodes[node_proposal.node_idx], *new_parent_node)
    return nodes


@njit
def prune(nodes: np.ndarray, node_proposal: NodeProposal):
    node = nodes[node_proposal.node_idx]
    # assert node["is_leaf"] == 0, str(node_proposal.node_idx)
    nodes[node["left"]]["active"] = 0
    nodes[node["right"]]["active"] = 0
    node["is_leaf"] = 1
    return nodes


@njit
def change(nodes: np.ndarray, node_proposal: NodeProposal):
    node = nodes[node_proposal.node_idx]
    node["feature_idx"] = node_proposal.new_feature_idx
    node["threshold"] = node_proposal.new_threshold
    return nodes


@njit
def get_tree_proposal(
    nodes: np.ndarray,
    bounds: np.ndarray,
    feat_types,
    params: "BARKTrainParamsNumba",
) -> tuple[np.ndarray, float]:
    node_proposal = NodeProposal()
    # numba doesn't support weighted choice
    proposal_idx = np.searchsorted(np.cumsum(params.proposal_weights), np.random.rand())
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

    node_proposal.node_idx = np.random.choice(valid_nodes)

    node_proposal.prev_feature_idx = nodes[node_proposal.node_idx]["feature_idx"]
    node_proposal.prev_threshold = nodes[node_proposal.node_idx]["threshold"]

    if (
        proposal_type == TreeProposalEnum.Grow
        or proposal_type == TreeProposalEnum.Change
    ):
        subspace = get_node_subspace(nodes, node_proposal.node_idx, bounds, feat_types)
        (
            node_proposal.new_feature_idx,
            node_proposal.new_threshold,
        ) = sample_splitting_rule(subspace, feat_types)

        # catch invalid splits for discrete features
        if (
            node_proposal.new_threshold == 0
            and feat_types[node_proposal.new_feature_idx] == FeatureTypeEnum.Cat.value
        ):
            return nodes, -np.inf

        if (
            node_proposal.new_threshold == subspace[node_proposal.new_feature_idx, 1]
            and feat_types[node_proposal.new_feature_idx] == FeatureTypeEnum.Int.value
        ):
            return nodes, -np.inf

    # I have no idea why this happens
    # tree_q_ratio has some side effect with the node_idx

    temp = node_proposal.node_idx
    log_q_ratio = tree_q_ratio(nodes, proposal_type, node_proposal)
    node_proposal.node_idx = temp

    log_prior_ratio = tree_prior_ratio(nodes, proposal_type, node_proposal, params)

    new_nodes = nodes.copy()
    if proposal_type == TreeProposalEnum.Grow:
        new_nodes = grow(new_nodes, node_proposal)
    elif proposal_type == TreeProposalEnum.Prune:
        new_nodes = prune(new_nodes, node_proposal)
    else:
        new_nodes = change(new_nodes, node_proposal)

    log_q_prior = log_q_ratio + log_prior_ratio
    return new_nodes, log_q_prior
