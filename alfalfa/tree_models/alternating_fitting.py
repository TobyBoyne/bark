import torch
import gpytorch as gpy
from tqdm import tqdm

from .forest import Node, AlfalfaTree
from .tree_kernels import ATGP, AFGP, AlfalfaGP
from ..utils.logger import Timer

N_ITERS = 10
N_TREE_PER_ITER = 3
N_GP_PER_ITER = 10

timer = Timer()

def all_cat_combinations(n):
    # only need to go up to n // 2
    # beyond this, it's equivalent
    for r in range(1, n//2 + 1):
        for comb in torch.combinations(torch.arange(n), r=r):
            yield comb

def _fit_decision_node(
    node: Node,
    tree: AlfalfaTree,
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    leaves = tree.root(x)
    in_node = node.contains_leaves(leaves)
    # reduced_x = x[,:]
    min_loss_var_idx = node.var_idx
    min_loss_t = node.threshold
    min_loss = torch.inf
    for var_idx, num_cats in enumerate(tree.var_is_cat):
        node.var_idx = var_idx
        if num_cats:
            # fit categorical variables
            for comb in all_cat_combinations(num_cats):
                node.threshold = comb
                output = model(x)

                # find threshold that minimises the MLL of the GPs
                loss = -mll(output, y)
                if loss < min_loss:
                    min_loss = loss
                    min_loss_t = comb
                    min_loss_var_idx = var_idx

        else:
            # var = torch.sort(var).values
            var = x[in_node, var_idx]
            for t in var[::2]:
                node.threshold = t
                output = model(x)

                # find threshold that minimises the MLL of the GPs
                loss = -mll(output, y)
                if loss < min_loss:
                    min_loss = loss
                    min_loss_t = t
                    min_loss_var_idx = var_idx
    node.threshold = min_loss_t
    node.var_idx = min_loss_var_idx


def fit_tree(
    tree: AlfalfaTree,
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    """Fit a decision tree using a GP marginal likelihood loss.

    This implements one step of the Tree Alternating Optimsation (TAO)
    algorithm to non-greedily fit a decision tree.
    """
    with torch.no_grad():
        # starting from the deepest decision nodes, iterate through every
        # layer of the tree, ending at the root
        for i in range(N_TREE_PER_ITER):
            for d in range(tree.depth - 1, -1, -1):
                nodes = tree.nodes_by_depth[d]
                for node in nodes:
                    _fit_decision_node(node, tree, x, y, model, mll)


def fit_forest(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    """Fit a forest of decision trees using a GP marginal likelihood loss."""
    if isinstance(model, ATGP):
        tree = model.tree
        fit_tree(tree, x, y, model, mll)
    elif isinstance(model, AFGP):
        for tree in model.forest.trees:
            fit_tree(tree, x, y, model, mll)


def fit_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    """Fit the (non-tree) hyperparameters of a Tree GP."""

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    training_iter = 10
    for i in (pbar := tqdm(range(training_iter))):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()

        pbar.set_description(
            f"Loss: {loss.item():.3f}, noise: {model.likelihood.noise.item():.3f}"
        )
        optimizer.step()


def fit_tree_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    """Fit a Tree GP.

    Alternately fits the tree and non-tree hyperparameters of the GP."""

    model.train()
    model.likelihood.train()

    for i in range(10):
        fit_forest(x, y, model, mll)
        fit_gp(x, y, model, mll)

        print(timer)
