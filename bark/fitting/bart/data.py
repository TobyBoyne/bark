"""Wrapper for data"""
import numpy as np
from beartype.cave import IntType
from beartype.typing import Any, Optional
from bofire.data_models.domain.api import Domain
from jaxtyping import Bool, Int, Shaped

from bark.utils.domain import get_cat_idx_from_domain

from ...forest import BARKNode, BARKTree, DecisionNode


class BARKData:
    """Contains the data used during training, as well as helper functions for
    e.g. unique values at nodes"""

    def __init__(self, domain: Domain, X: Shaped[np.ndarray, "N D"]):
        self.domain = domain
        self.X = X
        self.cat_idx = get_cat_idx_from_domain(domain)

    def get_init_prior(self):
        def _prior(node: DecisionNode):
            var_idx, threshold = self.sample_splitting_rule(node.tree, node)
            node.var_idx = var_idx
            node.threshold = threshold

        return _prior

    def sample_splitting_rule(
        self, tree: BARKTree, node: BARKNode, rng: np.random.Generator
    ) -> Optional[tuple[IntType, Any]]:
        x_index = self.get_x_index(tree, node)
        valid_features = self.valid_split_features(x_index)
        if not valid_features.size:
            # no valid splits to be made
            return
        var_idx = rng.choice(valid_features)

        valid_values = self.unique_split_values(x_index, var_idx)
        # TODO: should endpoints be excluded for continuous variables?
        threshold = rng.choice(valid_values)
        return var_idx, threshold

    def get_x_index(self, tree: BARKTree, node: BARKNode) -> Bool[np.ndarray, "N"]:
        """Get the index of datapoints that pass through the given node"""
        active_leaves = tree(self.X)
        return node.contains_leaves(active_leaves)

    def valid_split_features(
        self, x_index: Shaped[np.ndarray, "N"]
    ) -> Int[np.ndarray, "..."]:
        valid = [
            i
            for i in range(len(self.domain.inputs))
            if len(self.unique_split_values(x_index, i)) >= 1
        ]
        return np.array(valid, dtype=int)

    def unique_split_values(
        self, x_index: Shaped[np.ndarray, "N"], var_idx: IntType | int
    ) -> Shaped[np.ndarray, "unique"]:
        """
        x_index is shape (N,), where it is true if the x value reaches a leaf"""
        x = self.X[x_index, var_idx]
        if var_idx in self.cat_idx:
            return np.unique(x)
        else:
            return np.unique(x)[:-1]
