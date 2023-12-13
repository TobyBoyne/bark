import torch
from alfalfa.tree_models.forest import AlfalfaTree
import numpy as np

np.random.seed(42)
x_0 = torch.rand(size=(100,))
# x_0 = np.random.randint(low=0, high=10, size=(10,))
x_1 = torch.randint(low=0, high=2, size=(100,))
print(x_1)
X = torch.stack((x_0, x_1), axis=1)

y = torch.where(x_1 < 50, 0.0, 10.0)

tree = AlfalfaTree(depth=3)
tree.initialise_tree([0, 2])
print(tree.root(X))


def test_categorical_tree():
    tree = AlfalfaTree(depth=3)
    tree.initialise_tree(var_is_cat=[0, 5])
    print(tree)

test_categorical_tree()