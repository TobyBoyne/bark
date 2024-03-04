"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""
from typing import Optional

import lightgbm as lgb
import numpy as np

from ..forest import AlfalfaForest, AlfalfaTree, DecisionNode, LeafNode


def fit_lgbm_forest(
    train_x: np.ndarray, train_y: np.ndarray, params: Optional[dict] = None
) -> lgb.Booster:
    if params is None:
        params = {"max_depth": 3, "min_data_in_leaf": 1, "verbose": -1}

    return lgb.train(
        params,
        lgb.Dataset(train_x, train_y),
        num_boost_round=50,
    )


def lgbm_to_alfalfa_forest(tree_model: lgb.Booster) -> AlfalfaForest:
    all_trees = tree_model.dump_model()["tree_info"]

    def get_subtree(node_dict):
        if "leaf_index" in node_dict:
            return LeafNode()
        else:
            var_idx = node_dict["split_feature"]
            threshold = node_dict["threshold"]
            return DecisionNode(
                var_idx=var_idx,
                threshold=threshold,
                left=get_subtree(node_dict["left_child"]),
                right=get_subtree(node_dict["right_child"]),
            )

    trees = [
        AlfalfaTree(root=get_subtree(tree_dict["tree_structure"]))
        for tree_dict in all_trees
    ]
    forest = AlfalfaForest(trees=trees)
    return forest
