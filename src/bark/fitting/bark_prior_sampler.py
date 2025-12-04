"""Sample from the BARK prior."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

import bark.forest as forest
from bark import types
from bark.enums import NodeState
from bark.fitting.tree_proposals import grow, sample_splitting_rule
from bark.fitting.tree_traversal import get_node_subspace


def sample_forest(
    m: int,
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> types.Tree:
    trees = forest.create_empty_forest(m, max_depth=6)
    keys = jax.random.split(key, num=m)
    return jax.vmap(sample_tree, in_axes=(0, None, None, None, 0))(
        trees,
        bounds,
        feat_types,
        params,
        keys,
    )


def sample_tree(
    tree: types.Tree,
    bounds: types.BoundsT,
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
) -> types.Tree:
    def split_node(node_idx: Int[Array, ""], val: tuple[types.Tree, jax.Array]):
        tree, key = val
        key, subkey = jax.random.split(key)
        subspace = get_node_subspace(tree, node_idx, bounds, feat_types)
        new_feature_idx, new_threshold = sample_splitting_rule(
            subspace, feat_types, subkey
        )
        new_tree = grow(tree, node_idx, new_feature_idx, new_threshold)

        key, subkey = jax.random.split(key)
        invalid_threshold = (new_threshold == subspace[:, new_feature_idx]).any(axis=-1)
        parent_is_leaf = tree.feature_idx[forest.parent(node_idx)] == NodeState.Leaf
        is_root_node = node_idx == 0
        cannot_split = invalid_threshold | (parent_is_leaf & ~is_root_node)
        split_prob = params.alpha * jnp.pow(1 + forest.depth(node_idx), -params.beta)
        accept = jax.random.uniform(subkey) + cannot_split < split_prob

        tree = jax.tree_util.tree_map(
            lambda nt, t: jnp.where(accept, nt, t), new_tree, tree
        )
        return (tree, key)

    max_nodes = tree.feature_idx.shape[-1]
    tree, _ = jax.lax.fori_loop(0, max_nodes // 2, split_node, init_val=(tree, key))

    return tree


def sample_noise_prior(
    params: types.BARKConfig,
    key: jax.Array,
) -> Float[Array, ""]:
    gamma_samples = jax.random.gamma(key, a=params.gamma_prior_shape)
    return (1 / gamma_samples) / params.gamma_prior_rate
