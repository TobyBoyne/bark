"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""
import lightgbm as lgb
import pandas as pd
from beartype.typing import Optional
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput

from ..forest import AlfalfaForest, AlfalfaTree, DecisionNode, LeafNode


def fit_lgbm_forest(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
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

    return lgb.train(
        params,
        dataset,
    )


def lgbm_to_alfalfa_forest(tree_model: lgb.Booster) -> AlfalfaForest:
    all_trees = tree_model.dump_model()["tree_info"]

    def get_subtree(node_dict):
        if "leaf_index" in node_dict:
            return LeafNode()
        else:
            var_idx = node_dict["split_feature"]
            # var_key = tree_model.feature_name()[var_idx]
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
    forest = AlfalfaForest(trees=trees, frozen=True)
    return forest
