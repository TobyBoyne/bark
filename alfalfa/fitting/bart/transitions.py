"""Processes for mutating trees"""
from alfalfa.tree_models.forest import DecisionNode, AlfalfaTree, LeafNode
from ...tree_models.tree_kernels import AlfalfaGP
from .tree_traversal import terminal_nodes, singly_internal_nodes
from .data import Data
from .params import BARTTrainParams
import abc
from typing import Literal
import torch
import numpy as np
import gpytorch as gpy


def propose_transition(data: Data, tree: AlfalfaTree, params: BARTTrainParams) -> "Transition":
    parent_of_leaf = tree.root.right
    child_direction = "left"
    return GrowTransition(tree, parent_of_leaf, child_direction, 0, 0.2)
    return ChangeTransition(tree, tree.root, torch.tensor(0), torch.tensor(0.5))



class Transition(abc.ABC):
    """Proposed tree transition"""
    def __init__(self, tree: AlfalfaTree):
        self.tree = tree

    @abc.abstractmethod
    def apply(self):
        pass

    @abc.abstractmethod
    def apply_inverse(self):
        self.inverse.apply()

    @property
    @abc.abstractmethod
    def inverse(self):
        pass

    @abc.abstractmethod
    def log_q_ratio(self):
        pass
    
    def log_likelihood_ratio(self, model: AlfalfaGP):
        mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        with self:
            output = model(model.train_inputs[0])
            likelihood_star = -mll(output, model.train_targets)

        output = model(model.train_inputs[0])
        likelihood = -mll(output, model.train_targets)

        return (likelihood_star - likelihood).item()


    @abc.abstractmethod
    def log_prior_ratio(self):
        pass


class GrowTransition(Transition):
    def __init__(self, tree: AlfalfaTree,
                #  parent_of_leaf: DecisionNode, 
                #  child_direction: Literal["left", "right"],
                 node: LeafNode,
                 var_idx: int, 
                 threshold: float
                 ):
        self.node = node

        self.new_var_idx = var_idx
        self.new_threshold = threshold

        super().__init__(tree)

    def apply(self):
        new_node = DecisionNode(var_idx=self.new_var_idx, threshold=self.new_threshold)
        self.node.parent.replace_child(self.node)
        # TODO: define some node.parent.replace_child()

    @property
    def inverse(self):
        self.node = LeafNode()

    @property
    def node(self):
        return getattr(self.parent_of_leaf, self.child_direction)
    
    @node.setter
    def node(self, value):
        return setattr(self.parent_of_leaf, self.child_direction, value)

    def log_q_ratio(self, data: Data):
        b = len(terminal_nodes(self.tree))
        x_index = data.get_x_index(self.tree, self.node)
        
        p_adj = len(data.valid_split_features(x_index))
        n_adj = len(data.unique_split_values(x_index, self.new_var_idx))
        with self:
            w_star = len(singly_internal_nodes(self.tree))
        
        return np.log(b * p_adj * n_adj / w_star)
    
    def log_prior_ratio(self, data: Data, alpha, beta):
        depth = self.node.depth

        leaf: LeafNode = getattr(self.parent_of_leaf, self.child_direction)
        x_index = data.get_x_index(self.tree, leaf)

        p_adj = len(data.valid_split_features(x_index))
        n_adj = len(data.unique_split_values(x_index, self.new_var_idx))


        return np.log(alpha) + \
            2 * np.log(1 - alpha / (2 + depth)**beta) + \
            - np.log((1 + depth)**beta - alpha) + \
            - np.log(p_adj * n_adj)

        
class PruneTransition(Transition):
    def __init__(self, tree, *args):
        super().__init__(tree)
    
class ChangeTransition(Transition):
    def __init__(self, tree: AlfalfaTree, node: DecisionNode, var_idx: torch.IntTensor, threshold: torch.IntTensor):
        self.node = node
        self.prev_var_idx = node.var_idx
        self.prev_threshold = node.threshold

        self.new_var_idx = var_idx
        self.new_threshold = threshold
        super().__init__(tree)

    def _mutate(self):
        self.node.var_idx = self.new_var_idx
        self.node.threshold = self.new_threshold

    def _mutate_inverse(self):
        self.node.var_idx = self.new_var_idx
        self.node.threshold = self.new_threshold

    def log_q_ratio(self):
        return 0