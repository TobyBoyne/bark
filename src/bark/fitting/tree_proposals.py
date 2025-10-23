import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

import bark.forest as forest
from bark import types
from bark.enums import FeatureTypeEnum, NodeState, TreeProposalEnum
from bark.fitting.tree_traversal import (
    get_node_subspace,
    singly_internal_nodes,
    terminal_nodes,
)
from bark.utils.bit_operations import sample_binary_mask


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


def prior_ratio_for_grow_proposal(
    depth: Int[Array, ""],
    params: types.BARKConfig,
) -> Float[Array, ""]:
    alpha = params.alpha
    beta = params.beta

    return (
        jnp.log(alpha)
        + 2 * jnp.log(1 - alpha / jnp.pow(2 + depth, beta))
        + -jnp.log(jnp.pow(1 + depth, beta) - alpha)
    )


def grow(
    tree: types.Tree,
    node_idx: Int[Array, ""],
    new_feature_idx: Int[Array, ""],
    new_threshold: Float[Array, ""],
) -> types.Tree:
    selected_idcs = [
        node_idx,
        forest.left(node_idx),
        forest.right(node_idx),
    ]

    feature_idx = tree.feature_idx.at[selected_idcs].set(
        jnp.array([new_feature_idx, NodeState.Leaf, NodeState.Leaf])
    )
    threshold = tree.threshold.at[selected_idcs].set(
        jnp.array([new_threshold, 0.0, 0.0])
    )

    return types.Tree(feature_idx=feature_idx, threshold=threshold)


def prune(
    tree: types.Tree,
    node_idx: Int[Array, ""],
):
    selected_idcs = [
        node_idx,
        forest.left(node_idx),
        forest.right(node_idx),
    ]

    feature_idx = tree.feature_idx.at[selected_idcs].set(
        jnp.array([NodeState.Leaf, NodeState.Inactive, NodeState.Inactive])
    )
    threshold = tree.threshold.at[selected_idcs].set(jnp.array([0.0, 0.0, 0.0]))

    return types.Tree(feature_idx=feature_idx, threshold=threshold)


def change(
    tree: types.Tree,
    node_idx: Int[Array, ""],
    new_feature_idx: Int[Array, ""],
    new_threshold: Float[Array, ""],
) -> types.Tree:
    feature_idx = tree.feature_idx.at[node_idx].set(new_feature_idx)
    threshold = tree.threshold.at[node_idx].set(new_threshold)

    return types.Tree(feature_idx=feature_idx, threshold=threshold)


def _get_grow_proposal(
    tree: types.Tree,
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[types.Tree, Float[Array, ""]]:
    valid_nodes = terminal_nodes(tree.feature_idx)

    node_idx = jax.random.choice(key, tree.feature_idx.shape[-1], p=valid_nodes)
    subspace = get_node_subspace(
        tree=tree,
        node_idx=node_idx,
        bounds=bounds,
        feat_types=feat_types,
    )

    key, subkey = jax.random.split(key)
    new_feature_idx, new_threshold = sample_splitting_rule(subspace, feat_types, subkey)

    invalid_threshold = (new_threshold == subspace[:, new_feature_idx]).any(axis=-1)

    # compute the MCMC transition ratio
    w_0 = terminal_nodes(tree.feature_idx).sum(axis=-1)
    new_tree = grow(tree, node_idx, new_feature_idx, new_threshold)
    w_1_star = singly_internal_nodes(new_tree.feature_idx).sum(axis=-1)

    tree_q_ratio = jnp.log(w_0) - jnp.log(w_1_star)
    depth = forest.depth(node_idx)
    tree_prior_ratio = prior_ratio_for_grow_proposal(depth, params)

    tree_q_prior_ratio = jnp.where(
        invalid_threshold, -jnp.inf, tree_q_ratio + tree_prior_ratio
    )
    return new_tree, tree_q_prior_ratio


def _get_prune_proposal(
    tree: types.Tree,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[types.Tree, Float[Array, ""]]:
    valid_nodes = singly_internal_nodes(tree.feature_idx)

    node_idx = jax.random.choice(key, tree.feature_idx.shape[-1], p=valid_nodes)

    # compute the MCMC transition ratio
    w_0_star = terminal_nodes(tree.feature_idx).sum() - 1
    w_1 = singly_internal_nodes(tree.feature_idx).sum()
    tree_q_ratio = np.log(w_1) - np.log(w_0_star)

    depth = forest.depth(node_idx)
    # tree prior ratio is just 1 / grow prior ratio
    tree_prior_ratio = -prior_ratio_for_grow_proposal(depth, params)
    tree_q_prior_ratio = tree_q_ratio + tree_prior_ratio

    new_tree = prune(tree, node_idx)

    return new_tree, tree_q_prior_ratio


def _get_change_proposal(
    tree: types.Tree,
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[types.Tree, Float[Array, ""]]:
    valid_nodes = singly_internal_nodes(tree.feature_idx)

    node_idx = jax.random.choice(key, tree.feature_idx.shape[-1], p=valid_nodes)
    subspace = get_node_subspace(
        tree=tree,
        node_idx=node_idx,
        bounds=bounds,
        feat_types=feat_types,
    )

    key, subkey = jax.random.split(key)
    new_feature_idx, new_threshold = sample_splitting_rule(subspace, feat_types, subkey)

    invalid_threshold = (new_threshold == subspace[:, new_feature_idx]).any(axis=-1)

    # compute the MCMC transition ratio
    w_0 = terminal_nodes(tree.feature_idx).sum(axis=-1)
    new_tree = change(tree, node_idx, new_feature_idx, new_threshold)
    w_1_star = singly_internal_nodes(new_tree.feature_idx).sum(axis=-1)

    tree_q_ratio = jnp.log(w_0) - jnp.log(w_1_star)
    tree_prior_ratio = 0.0

    tree_q_prior_ratio = jnp.where(
        invalid_threshold, -jnp.inf, tree_q_ratio + tree_prior_ratio
    )
    return new_tree, tree_q_prior_ratio


@jax.jit
def get_tree_proposal(
    tree: types.Tree,
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> tuple[
    Int[Array, " 2**max_depth"], Float[Array, " 2**max_depth"], Float[Array, ""]
]:
    keys = jax.random.split(key, 3)
    grow_proposal = _get_grow_proposal(tree, bounds, feat_types, params, keys[0])
    prune_proposal = _get_prune_proposal(tree, params, keys[1])
    change_proposal = _get_change_proposal(tree, bounds, feat_types, params, key)

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
        shape=tree.feature_idx.shape[:-1],
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
