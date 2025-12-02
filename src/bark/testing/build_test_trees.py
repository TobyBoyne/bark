from typing import Mapping, Sequence

import jax.numpy as jnp

from bark import forest, types
from bark.fitting.tree_proposals import grow

type TreeKey = tuple[int, float | int]
type TreeDict = Mapping[TreeKey, tuple[TreeDict | None, TreeDict | None]]


def build_trees_from_dict(tree_dicts: Sequence[TreeDict], max_depth: int = 6):
    m = len(tree_dicts)
    trees = forest.create_empty_forest(m=m, max_depth=max_depth)

    def _recurse_trees(t: types.Tree, d: TreeDict, node_idx) -> types.Tree:
        root = next(iter(d.keys()))
        feature_idx, threshold = root
        left, right = d[root]

        t = grow(t, node_idx, jnp.array(feature_idx), jnp.array(threshold))
        if left is not None:
            t = _recurse_trees(t, left, forest.left(node_idx))
        if right is not None:
            t = _recurse_trees(t, right, forest.right(node_idx))

        return t

    for tree_idx in range(m):
        tree = _recurse_trees(trees[tree_idx], tree_dicts[tree_idx], node_idx=0)
        trees = types.Tree(
            feature_idx=trees.feature_idx.at[tree_idx].set(tree.feature_idx),
            threshold=trees.threshold.at[tree_idx].set(tree.threshold),
        )

    return trees
