from dataclasses import dataclass

import torch
import gpytorch as gpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional

from .forest import Node, AlfalfaTree
from .tree_kernels import ATGP, AFGP, AlfalfaGP
from ..utils.logger import Timer, Logger
from ..utils.plots import plot_loss_logs


@dataclass
class AlternatingTrainParams:
    num_iterations: int = 3
    num_tree_per_iter: int = 3
    num_gp_per_iter: int = 20

    threshold_jitter: float = 0.1

timer = Timer()
logger = Logger()

def all_cat_combinations(n):
    # only need to go up to n // 2
    # beyond this, it's equivalent
    for r in range(1, n//2 + 1):
        for comb in torch.combinations(torch.arange(n), r=r):
            yield comb

def all_threshold_values(var: torch.Tensor):
    # two approaches here:
    # - get every midpoint (prone to overfitting?)
    # - linspace along axis
    # if not var.numel():
    #     return []
    midpoints = (var[:-1] + var[1:]) / 2
    thresholds = midpoints
    # thresholds = torch.linspace(var.min(), var.max(), var.shape[0])
    # thresholds = torch.concatenate((
    #     torch.tensor([-torch.inf]), thresholds
    # ))
    return thresholds[::2]


def _fit_decision_node(
    node: Node,
    tree: AlfalfaTree,
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
    train_params: AlternatingTrainParams,
):
    leaves = tree.root(x)
    in_node = node.contains_leaves(leaves)
    # reduced_x = x[,:]
    min_loss_var_idx = node.var_idx
    min_loss_t = node.threshold
    min_loss = torch.tensor(torch.inf)
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
            # check the pairwise mean values
            for t in all_threshold_values(var):
                node.threshold = t
                output = model(x)

                # find threshold that minimises the MLL of the GPs
                loss = -mll(output, y)
                if loss < min_loss:
                    min_loss = loss
                    min_loss_t = t
                    min_loss_var_idx = var_idx
    node.threshold = min_loss_t
    if train_params.threshold_jitter:
        node.threshold += train_params.threshold_jitter * torch.randn(())
    node.var_idx = min_loss_var_idx
    
    logger.log(min_loss=min_loss.item(), train_step="tree")


def fit_tree(
    tree: AlfalfaTree,
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
    train_params: AlternatingTrainParams,
):
    """Fit a decision tree using a GP marginal likelihood loss.

    This implements one step of the Tree Alternating Optimsation (TAO)
    algorithm to non-greedily fit a decision tree.
    """
    with torch.no_grad():
        # starting from the deepest decision nodes, iterate through every
        # layer of the tree, ending at the root
        for i in range(train_params.num_tree_per_iter):
            for d in range(tree.depth - 1, -1, -1):
                nodes = tree.nodes_by_depth[d]
                for node in nodes:
                    _fit_decision_node(node, tree, x, y, model, mll, train_params)


def fit_forest(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
    train_params: AlternatingTrainParams,
):
    """Fit a forest of decision trees using a GP marginal likelihood loss."""
    if isinstance(model, ATGP):
        tree = model.tree
        fit_tree(tree, x, y, model, mll, train_params)
    elif isinstance(model, AFGP):
        for tree in model.forest.trees:
            fit_tree(tree, x, y, model, mll, train_params)


def fit_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
    train_params: AlternatingTrainParams,
):
    """Fit the (non-tree) hyperparameters of a Tree GP."""

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    for i in (pbar := tqdm(range(train_params.num_gp_per_iter))):
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

        logger.log(min_loss=loss.item(), train_step="gp")


def fit_tree_gp(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlfalfaGP,
    mll: gpy.mlls.MarginalLogLikelihood,
    train_params: Optional[AlternatingTrainParams] = None,
    test_x: Optional[torch.Tensor] = None,
    test_y: Optional[torch.Tensor] = None,

):
    """Fit a Tree GP.

    Alternately fits the tree and non-tree hyperparameters of the GP."""

    if train_params is None:
        train_params = AlternatingTrainParams()

    model.train()
    model.likelihood.train()

    logging = not (test_x is None or test_y is None)
    def log_test_loss():
        if logging:
            model.eval()
            pred_dist = model.likelihood(model(test_x))
            logger.log(test_loss=gpy.metrics.negative_log_predictive_density(pred_dist, test_y).item())
            model.train()

    log_test_loss()

    for i in range(train_params.num_iterations):
        fit_gp(x, y, model, mll, train_params)
        log_test_loss()

        fit_forest(x, y, model, mll, train_params)
        log_test_loss()

    # fig, ax = plot_loss_logs(logger, "min_loss", "train_step", "test_loss")
    # fig.savefig("figs/alternating_training.png")