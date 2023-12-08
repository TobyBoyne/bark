import torch
import gpytorch as gpy
import matplotlib.pyplot as plt
from alternating_forest import AlternatingTree, AlternatingForest, Node
from alfalfa.utils.plots import plot_gp_1d


class AlternatingTreeKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree: AlternatingTree):
        super().__init__()
        self.tree = tree


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        # if diag:
        #     out = torch.ones(x1.shape[0])
        return self.tree.gram_matrix(x1, x2)
    

class AlternatingForestKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, forest: AlternatingForest):
        super().__init__()
        self._forest = forest


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        # if diag:
        #     out = torch.ones(x1.shape[0])
        return self._forest.gram_matrix(x1, x2)
    

class AlternatingGP(gpy.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super().__init__(train_inputs, train_targets, likelihood)
        self.tree_model = None
        self.mean_module = gpy.means.ZeroMean()

        # forest = AlternatingForest(depth=3, num_trees=10)
        tree = AlternatingTree(depth=2)
        tree_kernel = AlternatingTreeKernel(tree)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)
    
    @property
    def tree(self) -> AlternatingTree:
        return self.covar_module.base_kernel.tree


def _fit_decision_node(
        node: Node, 
        x: torch.tensor,
        y: torch.tensor,
        model: gpy.models.ExactGP,
        likelihood: gpy.likelihoods.Likelihood,
        mll: gpy.mlls.MarginalLogLikelihood):
    # TODO: enumerate through variables
    leaves = node(x)
    in_node = node.contains_leaves(leaves)
    # reduced_x = x[,:]
    var = torch.select(x, index=0, dim=1)
    min_loss_t = None
    min_loss = torch.inf
    for t in var:
        node.threshold = t
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        if loss < min_loss:
            min_loss = loss
            min_loss_t = t

    node.threshold = min_loss_t

def fit_tree(tree: AlternatingTree, x, y, 
             model: gpy.models.ExactGP,
            likelihood: gpy.likelihoods.Likelihood,
            mll: gpy.mlls.MarginalLogLikelihood
        ):
    
    with torch.no_grad():
        # starting from the deepest decision nodes, iterate through every
        # layer of the tree, ending at the root
        for d in range(tree.depth-1, -1, -1):
            nodes = tree.nodes_by_depth[d]
            for node in nodes:
                _fit_decision_node(node, x, y, model, likelihood, mll)


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.linspace(0, 1, 10).reshape((-1, 1))
    y = (5 * x * torch.sin(5*x)).flatten()

    likelihood = gpy.likelihoods.GaussianLikelihood()
    gp = AlternatingGP(x, y, likelihood)

    # gp.eval()
    # x_test = torch.linspace(0, 5, 50).reshape((-1, 1))

    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    output = gp(x)
    # Calc loss and backprop gradients
    loss = -mll(output, y)
    print(loss)
    print(gp.tree.root.threshold)

    fit_tree(gp.tree, x, y, gp, likelihood, mll)
    fit_tree(gp.tree, x, y, gp, likelihood, mll)
    fit_tree(gp.tree, x, y, gp, likelihood, mll)
    fit_tree(gp.tree, x, y, gp, likelihood, mll)
    output = gp(x)
    loss = -mll(output, y)
    print("> ", loss)
    gp.eval()
    print(gp.tree)
    plot_gp_1d(gp, likelihood, x, y)
    plt.show()