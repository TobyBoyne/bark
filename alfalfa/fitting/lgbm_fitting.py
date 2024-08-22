"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from beartype.typing import Optional
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput

from ..forest_numba import NODE_RECORD_DTYPE


def fit_lgbm_forest(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    domain: Domain,
    params: Optional[dict] = None,
) -> lgb.Booster:
    default_params = {
        "max_depth": 3,
        "min_data_in_leaf": 1,
        "verbose": -1,
        "num_boost_round": 50,
    }
    if params is None:
        params = {}

    params = {**default_params, **params}

    cat = domain.inputs.get_keys(includes=CategoricalInput)
    dataset = lgb.Dataset(train_x, train_y, categorical_feature=cat)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = lgb.train(
            params,
            dataset,
        )

    return booster


def lgbm_to_alfalfa_forest(tree_model: lgb.Booster) -> np.ndarray:
    all_trees = tree_model.dump_model()["tree_info"]
    forest = np.zeros((len(all_trees), 100), dtype=NODE_RECORD_DTYPE)

    def get_tree(tree_dict):
        stack = [(0, tree_dict)]
        out = np.zeros((100,), dtype=NODE_RECORD_DTYPE)
        next_inactive = 1
        while stack:
            node_idx, node_dict = stack.pop()
            if "leaf_index" in node_dict:
                out[node_idx] = (1, 0, 0, 0, 0, 0, 1)

            else:
                out[node_idx] = (
                    0,
                    node_dict["split_feature"],
                    node_dict["threshold"],
                    next_inactive,
                    next_inactive + 1,
                    0,
                    1,
                )
                stack.append((next_inactive, node_dict["left_child"]))
                stack.append((next_inactive + 1, node_dict["right_child"]))
                next_inactive += 2

        return out

    for i, tree_dict in enumerate(all_trees):
        forest[i] = get_tree(tree_dict["tree_structure"])

    return forest

    #     if "leaf_index" in node_dict:
    #         return LeafNode()
    #     else:
    #         var_idx = node_dict["split_feature"]
    #         threshold = node_dict["threshold"]
    #         # TODO: double check this:
    #         if node_dict["decision_type"] == "==":
    #             threshold = [int(threshold)]

    #         return DecisionNode(
    #             var_idx=var_idx,
    #             threshold=threshold,
    #             left=get_subtree(node_dict["left_child"]),
    #             right=get_subtree(node_dict["right_child"]),
    #         )

    # trees = [
    #     AlfalfaTree(root=get_subtree(tree_dict["tree_structure"]))
    #     for tree_dict in all_trees
    # ]
    # forest = AlfalfaForest(trees=trees, frozen=True)
    # return forest
