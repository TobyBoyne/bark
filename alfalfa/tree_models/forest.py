import torch
from typing import Optional
from operator import attrgetter

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
    
    def get_tree_height(self):
        return 0

    def structure_eq(self, other):
        return isinstance(other, Leaf)


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
    
    def get_tree_height(self):
        return 1 + max(self.left.get_tree_height(), self.right.get_tree_height())
    
    def extra_repr(self):
        return f"(x_{self.var_idx}<{self.threshold:.3f})"
    
    def get_extra_state(self):
        return {"threshold": self.threshold, "var_idx": self.var_idx}
    
    def set_extra_state(self, state):
        self.threshold = state["threshold"]
        self.var_idx = state["var_idx"]

    def structure_eq(self, other):
        if isinstance(other, Node):
            return (
                self.threshold == other.threshold and
                self.var_idx == other.var_idx and
                self.left.structure_eq(other.left) and 
                self.right.structure_eq(other.right)
            )
        return False

    
class AlfalfaTree(torch.nn.Module):
    def __init__(self, depth=3, root:Optional[Node]=None):
        super().__init__()
        if root:
            self.root = root
            self.depth = root.get_tree_height()
        else:
            self.root = Node.create_of_depth(depth)
            self.depth = depth

        self.nodes_by_depth = self._get_nodes_by_depth()
        self.register_load_state_dict_post_hook(self._prune_tree_hook)

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
        
    def _prune_tree_hook(self, module, incompatible_keys):
        """Post hook for load_state_dict to handle missing nodes.
        
        This transforms any nodes that are missing data to leaves, effectively
        'pruning' branches of the tree. This function is to be used as a pre-hook
        for torch.load_state_dict, and is registered in the constructor."""

        while incompatible_keys.missing_keys:
            key = incompatible_keys.missing_keys.pop()
            *parent_key, child, _ = key.split(".")
            parent_node = attrgetter(".".join(parent_key))(module)
            setattr(parent_node, child, Leaf())
                
    def structure_eq(self, other: "AlfalfaTree"):
        return self.root.structure_eq(other.root)


class AlfalfaForest(torch.nn.Module):
    def __init__(self, depth=None, num_trees=None, trees: Optional[list[AlfalfaTree]] = None):
        super().__init__()
        if trees:
            self.trees = torch.nn.ModuleList(trees)
            self.depth = max(tree.depth for tree in trees)
        else:
            self.trees = torch.nn.ModuleList([AlfalfaTree(depth) for _ in range(num_trees)])
            self.depth = depth

    def gram_matrix(self, x1: torch.tensor, x2: torch.tensor):
        x1_leaves = torch.stack([tree.root(x1) for tree in self.trees], dim=1)
        x2_leaves = torch.stack([tree.root(x2) for tree in self.trees], dim=1)

        sim_mat = torch.eq(x1_leaves[:, None, :], x2_leaves[None, :, :])
        sim_mat = 1 / len(self.trees) * torch.sum(sim_mat, dim=2)
        return sim_mat
    
    def get_extra_state(self):
        return {"depth": self.depth, "num_trees": len(self.trees)}
    
    def set_extra_state(self, state):
        if self.depth != state["depth"]:
            raise ValueError(f"Saved model has depth {state['depth']}.")
        
        if len(self.trees) != state["num_trees"]:
            raise ValueError(f"Saved model has {state['num_trees']} trees.")

    def structure_eq(self, other: "AlfalfaForest"):
        return all(tree.structure_eq(other_tree) for tree, other_tree in zip(self.trees, other.trees))