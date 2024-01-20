"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""
import lightgbm as lgb
import torch

from .forest import DecisionNode, AlfalfaTree, LeafNode, AlfalfaForest


def lgbm_to_alfalfa_forest(tree_model: lgb.Booster):
    all_trees = tree_model.dump_model()["tree_info"]

    def get_subtree(node_dict):
        if "leaf_index" in node_dict:
            return LeafNode()
        else:
            var_idx = node_dict["split_feature"]
            threshold = torch.tensor(node_dict["threshold"])
            return DecisionNode(
                var_idx=var_idx,
                threshold=threshold,
                left=get_subtree(node_dict["left_child"]),
                right=get_subtree(node_dict["right_child"])
            )

    trees = [AlfalfaTree(root=get_subtree(tree_dict["tree_structure"])) for tree_dict in all_trees]
    forest = AlfalfaForest(trees=trees)
    return forest