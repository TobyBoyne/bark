"""Processes for mutating trees"""
import abc

import numpy as np
import torch
from beartype.cave import IntType
from beartype.typing import Optional

from ...forest import AlfalfaTree, DecisionNode, LeafNode
from ...tree_kernels import AlfalfaGP
from ...utils.logger import Timer
from .data import Data
from .params import BARTTrainParams, TransitionEnum
from .tree_traversal import singly_internal_nodes, terminal_nodes

tree_timer = Timer()


def propose_transition(
    data: Data, tree: AlfalfaTree, params: BARTTrainParams
) -> "Transition":
    step_idx = np.random.choice(len(params.step_weights), p=params.step_weights)

    if step_idx == TransitionEnum.GROW.value:
        leaf_nodes = terminal_nodes(tree)
        if not leaf_nodes:
            return
        cur_node = np.random.choice(leaf_nodes)

        new_node_data = data.sample_splitting_rule(tree, cur_node)
        if new_node_data is None:
            return

        new_node = DecisionNode(*new_node_data)
        return GrowTransition(tree, cur_node, new_node)

    elif step_idx == TransitionEnum.PRUNE.value:
        internal_nodes = singly_internal_nodes(tree)
        if not internal_nodes:
            return
        cur_node = np.random.choice(internal_nodes)

        return PruneTransition(tree, cur_node)

    elif step_idx == TransitionEnum.CHANGE.value:
        internal_nodes = singly_internal_nodes(tree)
        if not internal_nodes:
            return
        cur_node = np.random.choice(internal_nodes)
        new_node_data = data.sample_splitting_rule(tree, cur_node)
        if new_node_data is None:
            return

        return ChangeTransition(tree, cur_node, *new_node_data)

    else:
        raise ValueError("Unrecognised transition step")


def tree_acceptance_probability(
    data: Data, model: AlfalfaGP, transition: "Transition", params: BARTTrainParams
):
    # P(INVERSE_METHOD) / P(METHOD)
    # Not necessary as long as P(GROW) == P(PRUNE)
    with tree_timer("q_ratio"):
        q_ratio = transition.log_q_ratio(data)
    if np.isneginf(q_ratio):
        # no valid ways to perform the operation
        # e.g. there are no valid splitting rules for a given node
        return -np.inf
    with tree_timer("ll_ratio"):
        likelihood_ratio = transition.log_likelihood_ratio(model)
    with tree_timer("prior_ratio"):
        prior_ratio = transition.log_prior_ratio(data, params.alpha, params.beta)

    return min(q_ratio + likelihood_ratio + prior_ratio, 0.0)


def my_mll(model: AlfalfaGP):
    covar = model.covar_module(model.train_inputs[0])
    noise_covar = model.likelihood._shaped_noise_covar(torch.Size([covar.shape[-1]]))
    full_covar = covar + noise_covar

    y = model.train_targets.reshape(-1, 1)

    data_fit = y.T @ torch.linalg.solve(full_covar, y)
    complexity = torch.logdet(full_covar)
    return -data_fit - complexity


class Transition(abc.ABC):
    """Proposed tree transition"""

    def __init__(self, tree: AlfalfaTree):
        self.tree = tree

    @abc.abstractmethod
    def apply(self):
        pass

    def apply_inverse(self):
        self.inverse.apply()

    @property
    @abc.abstractmethod
    def inverse(self) -> "Transition":
        pass

    @abc.abstractmethod
    def log_q_ratio(self):
        pass

    def log_likelihood_ratio(self, model: AlfalfaGP):
        with self:
            likelihood_star = my_mll(model)

        # output = model(model.train_inputs[0])
        # likelihood = mll(output, model.train_targets)
        likelihood = my_mll(model)

        return (likelihood_star - likelihood).item()

    @abc.abstractmethod
    def log_prior_ratio(self):
        pass

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *args):
        self.apply_inverse()


class GrowTransition(Transition):
    def __init__(self, tree: AlfalfaTree, cur_node: LeafNode, new_node: DecisionNode):
        self.cur_node = cur_node
        self.new_node = new_node

        super().__init__(tree)

    def apply(self):
        self.cur_node.replace_self(self.new_node)

    @property
    def inverse(self):
        return PruneTransition(self.tree, self.new_node, self.cur_node)

    def log_q_ratio(self, data: Data):
        b = len(terminal_nodes(self.tree))
        x_index = data.get_x_index(self.tree, self.cur_node)

        p_adj = len(data.valid_split_features(x_index))
        n_adj = len(data.unique_split_values(x_index, self.new_node.var_idx))
        with self:
            w_star = len(singly_internal_nodes(self.tree))

        return np.log(b * p_adj * n_adj) - np.log(w_star)

    def log_prior_ratio(self, data: Data, alpha, beta):
        depth = self.cur_node.depth

        leaf = self.cur_node
        x_index = data.get_x_index(self.tree, leaf)

        p_adj = len(data.valid_split_features(x_index))
        n_adj = len(data.unique_split_values(x_index, self.new_node.var_idx))

        return (
            np.log(alpha)
            + 2 * np.log(1 - alpha / (2 + depth) ** beta)
            + -np.log((1 + depth) ** beta - alpha)
            + -np.log(p_adj * n_adj)
        )


class PruneTransition(Transition):
    def __init__(
        self, tree, cur_node: DecisionNode, new_node: Optional[LeafNode] = None
    ):
        self.cur_node = cur_node
        self.new_node = new_node if new_node is not None else LeafNode()

        super().__init__(tree)

    def apply(self):
        self.cur_node.replace_self(self.new_node)

    @property
    def inverse(self):
        return GrowTransition(self.tree, self.new_node, self.cur_node)

    def log_q_ratio(self, data: Data):
        b = len(terminal_nodes(self.tree))

        w = len(singly_internal_nodes(self.tree))

        with self:
            x_index = data.get_x_index(self.tree, self.new_node)
            p_adj_star = len(data.valid_split_features(x_index))
            n_adj_star = len(data.unique_split_values(x_index, self.cur_node.var_idx))

        # return np.log((b-1) * p_adj * n_adj / w_star)
        return np.log(w) - np.log((b - 1) * p_adj_star * n_adj_star)

    def log_prior_ratio(self, data: Data, alpha, beta):
        with self:
            grow_prior_ratio = self.inverse.log_prior_ratio(data, alpha, beta)
        return -grow_prior_ratio


class ChangeTransition(Transition):
    def __init__(
        self,
        tree: AlfalfaTree,
        node: DecisionNode,
        new_var_idx: IntType,
        new_threshold: float,
        cur_var_idx: Optional[IntType] = None,
        cur_threshold: Optional[float] = None,
    ):
        self.node = node
        self.cur_var_idx = cur_var_idx if cur_var_idx is not None else node.var_idx
        self.cur_threshold = (
            cur_threshold if cur_threshold is not None else node.threshold
        )

        self.new_var_idx = new_var_idx
        self.new_threshold = new_threshold
        super().__init__(tree)

    def apply(self):
        self.node.var_idx = self.new_var_idx
        self.node.threshold = self.new_threshold

    @property
    def inverse(self):
        return ChangeTransition(
            self.tree,
            self.node,
            self.cur_var_idx,
            self.cur_threshold,
            self.new_var_idx,
            self.new_threshold,
        )

    def log_q_ratio(self, data: Data):
        return 0
        # x_index = data.get_x_index(self.tree, self.node)

        # with self:
        #     n_adj_star = len(data.unique_split_values(x_index, self.new_var_idx))
        # n_adj = len(data.unique_split_values(x_index, self.cur_var_idk))
        # return np.log(n_adj_star) - np.log(n_adj)

    def log_prior_ratio(self, *args):
        return 0
