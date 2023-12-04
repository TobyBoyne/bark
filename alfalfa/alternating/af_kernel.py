import torch
import gpytorch as gpy
from alternating_forest import AlternatingTree


class AlternatingTreeKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree: AlternatingTree):
        super().__init__()
        self._tree = tree


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        # if diag:
        #     out = torch.ones(x1.shape[0])
        return self._tree.gram_matrix(x1, x2)
    
if __name__ == "__main__":
    tree = AlternatingTree(depth=3)
    kernel = AlternatingTreeKernel(tree)

    test_x = torch.linspace(-1, 1, 10)
    print(kernel(test_x, test_x).evaluate())
