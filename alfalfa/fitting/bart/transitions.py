"""Processes for mutating trees"""
from alfalfa.tree_models.forest import Node, AlfalfaTree, Leaf
import abc
from typing import Literal
import torch

def propose_transition(tree: AlfalfaTree, probs: list[float]=None):
    # return GrowTransition(tree, tree.root, "left")
    return ChangeTransition(tree, tree.root, torch.tensor(0), torch.tensor(0.5))

class Transition(abc.ABC):
    """Context manager for proposed tree transition"""
    def __init__(self, tree: AlfalfaTree):
        self._apply_transition = False
        self._tree = tree

    @abc.abstractmethod
    def _mutate(self):
        pass

    @abc.abstractmethod
    def _mutate_inverse(self):
        pass

    def log_q_ratio(self):
        return self.log_q_inverse() - self.log_q()

    def log_q(self):
        """The transition probability log q(T, T*)"""
        raise NotImplementedError

    def log_q_inverse(self):
        raise NotImplementedError

    def __enter__(self):
        self._mutate()    
    
    def __exit__(self, *args):
        if not self._apply_transition:
            # revert change on exit if not accepted
            self._mutate_inverse()

    def accept(self):
        self._apply_transition = True


class GrowTransition(Transition):
    def __init__(self, tree: AlfalfaTree, parent_of_leaf: Node, child_direction: Literal["left", "right"]):
        self.parent_of_leaf = parent_of_leaf
        self.child_direction = child_direction
        if child_direction not in ("left", "right"):
            raise ValueError("Child direction must be in ('left', 'right')")
        super().__init__(tree)

    def _mutate(self):
        new_node = Node()
        setattr(self.parent_of_leaf, self.child_direction, new_node)

    def _mutate_inverse(self):
        setattr(self.parent_of_leaf, self.child_direction, Leaf())

    def q(self):
        return 
    
    def q_inverse(self):
        return
    
class ChangeTransition(Transition):
    def __init__(self, tree: AlfalfaTree, node: Node, var_idx: torch.Tensor[int], threshold: torch.Tensor[int]):
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