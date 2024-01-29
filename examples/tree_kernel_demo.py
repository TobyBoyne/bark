from alfalfa.tree_models import AlfalfaTree, DecisionNode
from alfalfa.tree_models.tree_kernels import AlfalfaTreeModelKernel
import torch
import matplotlib.pyplot as plt
from alfalfa.leaf_gp.space import Space

tree = AlfalfaTree(
    root=DecisionNode(0, 0.5, left=DecisionNode(
        0, 0.25
        )
    )
)
tree.initialise(Space([[0.0, 1.0]]))
kernel = AlfalfaTreeModelKernel(tree)

x = torch.linspace(0, 1, 50).reshape(-1, 1)
plt.plot(x, kernel(x, torch.tensor([[0.4]])))
plt.show()