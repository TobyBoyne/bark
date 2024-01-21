"""Wrapper for data"""
import torch
import numpy as np
from ...leaf_gp.space import Space
from ...tree_models.forest import AlfalfaTree, DecisionNode



class Data:
    def __init__(self, space: Space, X: np.ndarray):
        self.space = space
        self.X = X # (N, D)

    def get_x_index(self, tree: AlfalfaTree, node: DecisionNode):
        """Get the index of datapoints that pass through the given node"""
        active_leaves = tree.root(self.X)
        return node.contains_leaves(active_leaves)


    # def sample_rule_prior(self, x_index: torch.BoolTensor):
    #     var_idx = torch.randint(len(self.space))
    #     threshold = torch.randperm

    def valid_split_features(self, x_index: np.ndarray):
        var_idxs = np.arange(len(self.space))
        valid = [len(self.unique_split_values(x_index, i)) >= 2 for i in range(len(self.space))]
        return var_idxs[valid]

    def unique_split_values(self, x_index: np.ndarray, var_idx: int):
        """
        x_index is shape (N,), where it is true if the x value reaches a leaf"""
        x = self.X[x_index, var_idx]
        return np.unique(x)