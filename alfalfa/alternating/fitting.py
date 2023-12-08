import torch
import gpytorch as gpy
from tqdm import tqdm

from .af_kernel import Node, AlternatingTree, ATGP

N_ITERS = 10
N_TREE_PER_ITER = 10
N_GP_PER_ITER = 10

def _fit_decision_node(
    node: Node,
    x: torch.tensor,
    y: torch.tensor,
    model: gpy.models.ExactGP,
    likelihood: gpy.likelihoods.Likelihood,
    mll: gpy.mlls.MarginalLogLikelihood,
):
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

        # find threshold that minimises the MLL of the GPs
        loss = -mll(output, y)
        if loss < min_loss:
            min_loss = loss
            min_loss_t = t

    node.threshold = min_loss_t


def fit_tree(
    x: torch.tensor,
    y: torch.tensor,
    model: ATGP,
    likelihood: gpy.likelihoods.Likelihood,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    """Fit a decision tree using a GP marginal likelihood loss.
    
    This implements one step of the Tree Alternating Optimsation (TAO) 
    algorithm to non-greedily fit a decision tree.
    """
    with torch.no_grad():
        # starting from the deepest decision nodes, iterate through every
        # layer of the tree, ending at the root
        tree = model.tree
        for i in range(N_TREE_PER_ITER):
            for d in range(tree.depth - 1, -1, -1):
                nodes = tree.nodes_by_depth[d]
                for node in nodes:
                    _fit_decision_node(node, x, y, model, likelihood, mll)

def fit_gp(x: torch.tensor, y:torch.tensor, model: ATGP,
    likelihood: gpy.likelihoods.Likelihood,
    mll: gpy.mlls.MarginalLogLikelihood):
    """Fit the (non-tree) hyperparameters of a Tree GP."""
        
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    training_iter = 10
    for i in (pbar := tqdm(range(training_iter))):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        # print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}"
        #       f"  \tnoise: {model.likelihood.noise.item():.3f}"
        # )
        pbar.set_description(f"Loss: {loss.item():.3f}, noise: {model.likelihood.noise.item():.3f}")
        optimizer.step()

def fit_tree_gp(
    x: torch.tensor,
    y: torch.tensor,
    model: ATGP,
    likelihood: gpy.likelihoods.Likelihood,
    mll: gpy.mlls.MarginalLogLikelihood,
):
    """Fit a Tree GP.
    
    Alternately fits the tree and non-tree hyperparameters of the GP."""

    # for i in (pbar := tqdm(range(10))):
    for i in range(10):
        fit_tree(x, y, model, likelihood,mll)
        fit_gp(x, y, model, likelihood,mll)


