"""Sample from the BARK prior."""

import numpy as np

from bark.fitting.tree_proposals import (
    NodeProposal,
    _get_two_inactive_nodes,
    grow,
    sample_splitting_rule,
)
from bark.fitting.tree_traversal import get_node_subspace
from bark.forest import FeatureTypeEnum, create_empty_forest


def _sample_single_forest(
    m: int,
    bounds: np.ndarray,
    feat_types: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
):
    forest = create_empty_forest(m)

    for j in range(m):
        tree = forest[j, :]
        node_stack = [0]
        while node_stack:
            node_proposal = NodeProposal()
            node_proposal.node_idx = node_stack.pop()

            depth = tree[node_proposal.node_idx]["depth"]
            if rng.uniform() > alpha * (1 + depth) ** (-beta):
                continue

            subspace = get_node_subspace(
                tree, node_proposal.node_idx, bounds, feat_types
            )

            (
                node_proposal.new_feature_idx,
                node_proposal.new_threshold,
            ) = sample_splitting_rule(subspace, feat_types)

            if (
                node_proposal.new_threshold == 0
                and feat_types[node_proposal.new_feature_idx]
                == FeatureTypeEnum.Cat.value
            ):
                continue

            if (
                node_proposal.new_threshold
                == subspace[node_proposal.new_feature_idx, 1]
                and feat_types[node_proposal.new_feature_idx]
                == FeatureTypeEnum.Int.value
            ):
                continue

            left, right = _get_two_inactive_nodes(tree)
            tree = grow(tree, node_proposal)
            node_stack.append(left)
            node_stack.append(right)

    return forest


def sample_forest_prior(
    m: int,
    bounds: np.ndarray,
    feat_types: np.ndarray,
    alpha: float,
    beta: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    forests = [
        _sample_single_forest(m, bounds, feat_types, alpha, beta, rng)
        for _ in range(num_samples)
    ]
    return np.array(forests)


def sample_noise_prior(
    gamma_shape: float,
    gamma_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> float:
    return rng.gamma(shape=gamma_shape, scale=1 / gamma_rate, size=(num_samples,))
