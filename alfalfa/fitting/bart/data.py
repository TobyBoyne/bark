"""Wrapper for data"""
import torch
from ...leaf_gp.space import Space
from ...tree_models.forest import AlfalfaTree, Node



class Data:
    def __init__(self, space: Space, X: torch.Tensor):
        self.space = space
        self.X = X # (N, D)

    def get_x_index(self, tree: AlfalfaTree, node: Node):
        """Get the index of datapoints that pass through the given node"""
        active_leaves = tree.root(self.X)
        return torch.eq(active_leaves, node.child_leaves)


    def sample_rule_prior(self, x_index: torch.BoolTensor):
        var_idx = torch.randint(len(self.space))
        threshold = torch.randperm

    def valid_split_features(self, x_index: torch.BoolTensor):
        return torch.tensor(
            [
                len(self.unique_split_values(x_index, i)) >= 2
                for i in range(len(self.space))
            ]
        )            


    def unique_split_values(self, x_index: torch.BoolTensor, var_idx: int):
        """
        x_index is shape (N,), where it is true if the x value reaches a leaf"""
        x = self.X[x_index, var_idx]
        return torch.unique(x)