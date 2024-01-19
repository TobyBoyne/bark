"""Processes for mutating trees"""
from alfalfa.tree_models.forest import Node, AlfalfaTree, Leaf
from ...tree_models.tree_kernels import AlfalfaGP
from .tree_traversal import terminal_nodes, singly_internal_nodes
from .data import Data
from .params import BARTTrainParams
import abc
from typing import Literal
import torch
import gpytorch as gpy


def propose_transition(data: Data, tree: AlfalfaTree, params: BARTTrainParams) -> "Transition":
    parent_of_leaf = tree.root.right
    child_direction = "left"
    return GrowTransition(tree, parent_of_leaf, child_direction, torch.tensor(0), torch.tensor(0.2))
    return ChangeTransition(tree, tree.root, torch.tensor(0), torch.tensor(0.5))



class Transition(abc.ABC):
    """Context manager for proposed tree transition"""
    def __init__(self, tree: AlfalfaTree):
        self._apply_transition = False
        self.tree = tree

    @abc.abstractmethod
    def _mutate(self):
        pass

    @abc.abstractmethod
    def _mutate_inverse(self):
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

        return likelihood_star - likelihood


    @abc.abstractmethod
    def log_prior_ratio(self):
        pass

    def __enter__(self):
        self._mutate()    
    
    def __exit__(self, *args):
        if not self._apply_transition:
            # revert change on exit if not accepted
            self._mutate_inverse()

    def accept(self):
        self._apply_transition = True


class GrowTransition(Transition):
    def __init__(self, tree: AlfalfaTree,
                 parent_of_leaf: Node, 
                 child_direction: Literal["left", "right"],
                 var_idx: torch.IntTensor, 
                 threshold: torch.Tensor
                 ):
        self.parent_of_leaf = parent_of_leaf
        self.child_direction = child_direction
        if child_direction not in ("left", "right"):
            raise ValueError("Child direction must be in ('left', 'right')")

        self.new_var_idx = var_idx
        self.new_threshold = threshold

        super().__init__(tree)

    def _mutate(self):
        new_node = Node(var_idx=self.new_var_idx, threshold=self.new_threshold)
        self.node = new_node

    def _mutate_inverse(self):
        self.node = Leaf()

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
        
        return torch.log(torch.tensor(b * p_adj * n_adj / w_star))
    
    def log_prior_ratio(self, data: Data, alpha, beta):
        depth = self.node.depth

        leaf: Leaf = getattr(self.parent_of_leaf, self.child_direction)
        x_index = data.get_x_index(self.tree, leaf)

        p_adj = len(data.valid_split_features(x_index))
        n_adj = len(data.unique_split_values(x_index, self.new_var_idx))


        return torch.log(alpha) + \
            2 * torch.log(1 - alpha / torch.pow(2 + depth, beta)) + \
            - torch.log(torch.pow(1 + depth) - alpha) + \
            - torch.log(p_adj * n_adj)

        

    
class ChangeTransition(Transition):
    def __init__(self, tree: AlfalfaTree, node: Node, var_idx: torch.IntTensor, threshold: torch.IntTensor):
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