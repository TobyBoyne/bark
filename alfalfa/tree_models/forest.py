import torch
from typing import Any, Optional
import random

def _leaf_id_iter():
    i = 0
    while True:
        yield i
        i += 1

_leaf_id = _leaf_id_iter()

class Leaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf_id = next(_leaf_id)
        self.child_leaves = torch.tensor([self.leaf_id])
    
    def forward(self, _x):
        return torch.tensor(self.leaf_id)

class Node(torch.nn.Module):
    def __init__(self, var_idx=0, threshold=None, 
                 left: Optional["Node"] = None, 
                 right: Optional["Node"] = None):
        super().__init__()
        self.threshold = torch.randn(()) - 0.5 if threshold is None else threshold
        self.var_idx = var_idx
        self.left = Leaf() if left is None else left
        self.right = Leaf() if right is None else right

        self.child_leaves = torch.concat((self.left.child_leaves, self.right.child_leaves))

    def forward(self, x):
        var = torch.select(x, index=self.var_idx, dim=1)
        return torch.where(
            var < self.threshold, 
            self.left(x), 
            self.right(x),
        )
    
    @classmethod
    def create_of_depth(cls, d):
        """Create a node with depth d"""
        if d == 0:
            return Leaf()
        else:
            left = cls.create_of_depth(d-1)
            right = cls.create_of_depth(d-1)
            return Node(left=left, right=right)
        
    def contains_leaves(self, leaves: torch.tensor):
        return torch.isin(leaves, self.child_leaves)
    
    def extra_repr(self):
        return f"(x_{self.var_idx}<{self.threshold:.3f})"
    
    def get_extra_state(self):
        return {"threshold": self.threshold, "var_idx": self.var_idx}
    
    def set_extra_state(self, state):
        self.threshold = state["threshold"]
        self.var_idx = state["var_idx"]

    
class AlfalfaTree(torch.nn.Module):
    def __init__(self, depth=3, root:Optional[Node]=None):
        super().__init__()
        if root:
            self.root = root
        else:
            self.root = Node.create_of_depth(depth)
        self.nodes_by_depth = self._get_nodes_by_depth()
        self.depth = depth

    def _get_nodes_by_depth(self):
        nodes = [self.root]
        nodes_by_depth = {}
        depth = 0
        while nodes:
            nodes_by_depth[depth] = [*nodes]
            new_nodes = []
            for node in nodes:
                if not isinstance(node, Leaf):
                    new_nodes += [node.left, node.right]
            nodes = new_nodes
            depth += 1
        return nodes_by_depth
    
    @property
    def leaves(self):
        return self.nodes_by_depth[self.depth]
    
    def gram_matrix(self, x1: torch.tensor, x2: torch.tensor):
        x1_leaves = self.root(x1)
        x2_leaves = self.root(x2)

        sim_mat = torch.eq(x1_leaves[:, None], x2_leaves[None, :]).float()
        return sim_mat
    
    def get_extra_state(self):
        return {"depth": self.depth}
    
    def set_extra_state(self, state):
        if self.depth != state["depth"]:
            raise ValueError(f"Saved model has depth {state['depth']}.")



class AlfalfaForest(torch.nn.Module):
    def __init__(self, depth=3, num_trees=10, trees: Optional[list[AlfalfaTree]] = None):
        super().__init__()
        if trees:
            self.trees = torch.nn.ModuleList(trees)
        else:
            self.trees = torch.nn.ModuleList([AlfalfaTree(depth) for _ in range(num_trees)])
        self.depth = depth

    def gram_matrix(self, x1: torch.tensor, x2: torch.tensor):
        x1_leaves = torch.stack([tree.root(x1) for tree in self.trees], dim=1)
        x2_leaves = torch.stack([tree.root(x2) for tree in self.trees], dim=1)

        sim_mat = torch.eq(x1_leaves[:, None, :], x2_leaves[None, :, :])
        sim_mat = 1 / len(self.trees) * torch.sum(sim_mat, dim=2)
        return sim_mat
    
    # def get_extra_state(self):
    #     return {"depth": self.depth, "num_tress": len(self.trees)}
    
    # def set_extra_state(self, state):
    #     if self.depth != state["depth"]:
    #         raise ValueError(f"Saved model has depth {state['depth']}.")

