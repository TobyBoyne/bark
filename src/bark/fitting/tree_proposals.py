import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jaxtyping import Array, Float, Int
from numba import njit

import bark.forest as forest
from bark import types
from bark.enums import FeatureTypeEnum, NodeState, TreeProposalEnum
from bark.fitting.tree_traversal import (
    get_node_subspace,
    singly_internal_nodes,
    terminal_nodes,
)
from bark.utils.bit_operations import sample_binary_mask


# @jitclass(
#     [
#         ("node_idx", nb.uint32),
#         ("prev_feature_idx", nb.uint32),
#         ("prev_threshold", nb.float32),
#         ("new_feature_idx", nb.uint32),
#         ("new_threshold", nb.float32),
#     ]
# )
@struct.dataclass
class NodeProposal:
    node_idx: int
    prev_feature_idx: int
    prev_threshold: float
    new_feature_idx: int
    new_threshold: float


# @njit
# def _assign_node(target, is_leaf, feature_idx, threshold, active) -> None:
#     # numba requires individual assignment
#     # https://numba.discourse.group/t/assigning-to-numpy-structural-array-using-a-tuple-in-jitclass/549/6

#     target["is_leaf"] = is_leaf
#     target["feature_idx"] = feature_idx
#     target["threshold"] = threshold
#     target["active"] = active

#     # target["left"] = left
#     # target["right"] = right
#     # target["parent"] = parent
#     # target["depth"] = depth


# @jax.jit
def sample_splitting_rule(
    bounds: types.BoundsT, feat_types: types.FeatTypesT, key: jax.Array
) -> tuple[Int[Array, ""], Float[Array, ""] | Int[Array, ""]]:
    int_bounds = bounds.astype(jnp.uint64)
    keys = jax.random.split(key, num=4)
    feature_idx = jax.random.randint(keys[0], (), minval=0, maxval=bounds.shape[0])

    cat_sample = sample_binary_mask(int_bounds[feature_idx, 1], keys[1])
    int_sample = jax.random.randint(
        keys[2], (), int_bounds[feature_idx, 0] + 1, int_bounds[feature_idx, 1]
    )
    cont_sample = jax.random.uniform(
        keys[3], (), bounds[feature_idx, 0], bounds[feature_idx, 1]
    )

    condlist = [
        feat_types[feature_idx] == ft
        for ft in (FeatureTypeEnum.Cat, FeatureTypeEnum.Int, FeatureTypeEnum.Cont)
    ]
    threshold = jnp.select(condlist, [cat_sample, int_sample, cont_sample])

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


def prior_ratio_for_grow_proposal(
    depth: Float[Array, ""],
    params: types.BARKConfig,
):
    alpha = params.alpha
    beta = params.beta

    return (
        jnp.log(alpha)
        + 2 * jnp.log(1 - alpha / jnp.pow(2 + depth, beta))
        + -jnp.log(jnp.pow(1 + depth, beta) - alpha)
    )


@njit
def tree_prior_ratio(
    nodes: np.ndarray,
    proposal_type: int,
    node_proposal: NodeProposal,
    params: types.BARKConfig,
):
    alpha = params.alpha
    beta = params.beta
    depth = forest.depth(node_proposal.node_idx)

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


def grow(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    node_idx: Int[Array, ""],
    new_feature_idx: Int[Array, ""],
    new_threshold: Float[Array, ""],
) -> tuple[Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"]]:
    selected_idcs = [
        node_idx,
        forest.left(node_idx),
        forest.right(node_idx),
    ]

    feature_idx_tree = feature_idx_tree.at[selected_idcs].set(
        jnp.array([new_feature_idx, NodeState.Leaf, NodeState.Leaf])
    )
    threshold_tree = threshold_tree.at[selected_idcs].set(
        jnp.array([new_threshold, 0.0, 0.0])
    )

    return feature_idx_tree, threshold_tree


def prune(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    node_idx: Int[Array, ""],
):
    selected_idcs = [
        node_idx,
        forest.left(node_idx),
        forest.right(node_idx),
    ]

    feature_idx_tree = feature_idx_tree.at[selected_idcs].set(
        jnp.array([NodeState.Leaf, NodeState.Inactive, NodeState.Inactive])
    )
    threshold_tree = threshold_tree.at[selected_idcs].set(jnp.array([0.0, 0.0, 0.0]))

    return feature_idx_tree, threshold_tree


def change(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    node_idx: Int[Array, ""],
    new_feature_idx: Int[Array, ""],
    new_threshold: Float[Array, ""],
) -> tuple[Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"]]:
    feature_idx_tree = feature_idx_tree.at[node_idx].set(new_feature_idx)
    threshold_tree = threshold_tree.at[node_idx].set(new_threshold)

    return feature_idx_tree, threshold_tree


def _get_grow_proposal(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[
    Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"], Float[Array, ""]
]:
    valid_nodes = terminal_nodes(feature_idx_tree)

    node_idx = jax.random.choice(key, feature_idx_tree.shape[-1], p=valid_nodes)
    subspace = get_node_subspace(
        feature_idx_tree=feature_idx_tree,
        threshold_tree=threshold_tree,
        node_idx=node_idx,
        bounds=bounds,
        feat_types=feat_types,
    )

    key, subkey = jax.random.split(key)
    new_feature_idx, new_threshold = sample_splitting_rule(subspace, feat_types, subkey)

    invalid_threshold = (new_threshold == subspace[:, new_feature_idx]).any(axis=-1)

    # compute the MCMC transition ratio
    w_0 = terminal_nodes(feature_idx_tree).sum(axis=-1)
    new_feature_idx_tree, new_threshold_tree = grow(
        feature_idx_tree, threshold_tree, node_idx, new_feature_idx, new_threshold
    )
    w_1_star = singly_internal_nodes(new_feature_idx_tree).sum(axis=-1)

    tree_q_ratio = jnp.log(w_0) - jnp.log(w_1_star)
    depth = forest.depth(node_idx)
    tree_prior_ratio = prior_ratio_for_grow_proposal(depth, params)

    tree_q_prior_ratio = jnp.where(
        invalid_threshold, -jnp.inf, tree_q_ratio + tree_prior_ratio
    )
    return (new_feature_idx_tree, new_threshold_tree, tree_q_prior_ratio)


def _get_prune_proposal(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[
    Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"], Float[Array, ""]
]:
    valid_nodes = singly_internal_nodes(feature_idx_tree)

    node_idx = jax.random.choice(key, feature_idx_tree.shape[-1], p=valid_nodes)

    # compute the MCMC transition ratio
    w_0_star = terminal_nodes(feature_idx_tree).sum() - 1
    w_1 = singly_internal_nodes(feature_idx_tree).sum()
    tree_q_ratio = np.log(w_1) - np.log(w_0_star)

    depth = forest.depth(node_idx)
    # tree prior ratio is just 1 / grow prior ratio
    tree_prior_ratio = -prior_ratio_for_grow_proposal(depth, params)
    tree_q_prior_ratio = tree_q_ratio + tree_prior_ratio

    new_feature_idx_tree, new_threshold_tree = prune(
        feature_idx_tree, threshold_tree, node_idx
    )

    return (new_feature_idx_tree, new_threshold_tree, tree_q_prior_ratio)


def _get_change_proposal(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[
    Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"], Float[Array, ""]
]:
    valid_nodes = singly_internal_nodes(feature_idx_tree)

    node_idx = jax.random.choice(key, feature_idx_tree.shape[-1], p=valid_nodes)
    subspace = get_node_subspace(
        feature_idx_tree=feature_idx_tree,
        threshold_tree=threshold_tree,
        node_idx=node_idx,
        bounds=bounds,
        feat_types=feat_types,
    )

    key, subkey = jax.random.split(key)
    new_feature_idx, new_threshold = sample_splitting_rule(subspace, feat_types, subkey)

    invalid_threshold = (new_threshold == subspace[:, new_feature_idx]).any(axis=-1)

    # compute the MCMC transition ratio
    w_0 = terminal_nodes(feature_idx_tree).sum(axis=-1)
    new_feature_idx_tree, new_threshold_tree = grow(
        feature_idx_tree, threshold_tree, node_idx, new_feature_idx, new_threshold
    )
    w_1_star = singly_internal_nodes(new_feature_idx_tree).sum(axis=-1)

    tree_q_ratio = jnp.log(w_0) - jnp.log(w_1_star)
    tree_prior_ratio = 0.0

    tree_q_prior_ratio = jnp.where(
        invalid_threshold, -jnp.inf, tree_q_ratio + tree_prior_ratio
    )
    return (new_feature_idx_tree, new_threshold_tree, tree_q_prior_ratio)


@jax.jit
def get_tree_proposal(
    feature_idx_tree: Int[Array, " 2**max_depth"],
    threshold_tree: Float[Array, " 2**max_depth"],
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[
    Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"], Float[Array, ""]
]:
    keys = jax.random.split(key, 3)
    grow_proposal = _get_grow_proposal(
        feature_idx_tree, threshold_tree, bounds, feat_types, params, keys[0]
    )
    prune_proposal = _get_prune_proposal(
        feature_idx_tree, threshold_tree, params, keys[1]
    )
    change_proposal = _get_change_proposal(
        feature_idx_tree, threshold_tree, bounds, feat_types, params, key
    )

    (new_feature_idx_tree, new_threshold_tree, tree_q_prior_ratio) = (
        jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=-1),
            grow_proposal,
            prune_proposal,
            change_proposal,
        )
    )

    proposal_types = jnp.array(
        [
            TreeProposalEnum.Grow,
            TreeProposalEnum.Prune,
            TreeProposalEnum.Change,
        ]
    )
    proposal_type = jax.random.choice(
        key,
        a=proposal_types,
        p=params.proposal_weights,
        shape=feature_idx_tree.shape[:-1],
    )
    condlist = [
        proposal_type == tp
        for tp in (
            TreeProposalEnum.Grow,
            TreeProposalEnum.Prune,
            TreeProposalEnum.Change,
        )
    ]
    (new_feature_idx_tree, new_threshold_tree, tree_q_prior_ratio) = (
        jax.tree_util.tree_map(
            lambda *xs: jnp.select(condlist, xs),
            grow_proposal,
            prune_proposal,
            change_proposal,
        )
    )

    return new_feature_idx_tree, new_threshold_tree, tree_q_prior_ratio
