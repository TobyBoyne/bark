import torch
from typing import Optional

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
    
    def forward(self, _x):
        return torch.tensor(self.leaf_id)

class Node(torch.nn.Module):
    def __init__(self, var_idx=0, threshold=0.0, 
                 left: Optional["Node"] = None, 
                 right: Optional["Node"] = None):
        super().__init__()
        self.threshold = torch.nn.Parameter(data=torch.tensor(threshold), requires_grad=True)
        self.var_idx = var_idx
        self.left = Leaf() if left is None else left
        self.right = Leaf() if right is None else right

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

    
class AlternatingTree(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
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

        sim_mat = torch.eq(x1_leaves[:, None], x2_leaves[None, :])
        # sim_mat = torch.sum(sim_mat, dim=2)
        return sim_mat





class AlternatingForest(torch.nn.Module):
    pass

if __name__ == "__main__":
    # node1 = Node(var_idx=0, threshold=0.25)
    # node2 = Node(var_idx=0, threshold=0.75)
    # tree = Node(var_idx=0, threshold=0.5, left=node1, right=node2)
    
    # x = torch.tensor(torch.randn((100, 1)))
    # print(tree(x))

    tree = AlternatingTree(depth=3)
    print(tree.nodes_by_depth)