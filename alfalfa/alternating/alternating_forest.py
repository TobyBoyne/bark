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
    def __init__(self, var_idx, threshold, 
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
    
class AlternatingTree(torch.nn.Module):
    pass

class AlternatingForest(torch.nn.Module):
    pass

if __name__ == "__main__":
    node1 = Node(var_idx=0, threshold=0.25)
    node2 = Node(var_idx=0, threshold=0.75)
    tree = Node(var_idx=0, threshold=0.5, left=node1, right=node2)
    
    x = torch.tensor(torch.randn((100, 1)))
    print(tree(x))