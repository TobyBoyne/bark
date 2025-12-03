from typing import TYPE_CHECKING, Any, Callable

import gurobipy as gp
import numpy as np
from jaxtyping import Float

if TYPE_CHECKING:
    from bark.optimizer.mip_model import TreesMIPModel


class GurobiOptimizerModel(gp.Model):
    """An extended Gurobi model, complete with added variables."""

    _n_feat: int
    """Number of input features."""
    _cont_var_dict: dict[int, gp.Var]
    """Dictionary of ordinal (continuous and integer) input features."""
    _cat_var_dict: dict[int, dict[int, gp.Var]]
    """Dictionary of categorical input features."""

    _tree_models: dict[str, "TreesMIPModel"]

    _trees_set: set[str]

    _num_trees: Callable[[str], int]

    _leaves: Callable[[str, int], tuple[str, ...]]
    """Get the leaf encodings for a given (label, tree_index) pair."""

    _breakpoint_index: Any

    _breakpoints: Callable[[int], Any]

    _z_l: gp.tupledict[tuple[str, int, str], gp.Var]
    """Binary leaf variables, indexed by (label, tree_index, leaf_encoding)."""

    _y: Any

    _mu_coeff: Float[np.ndarray, "num_samples num_data"]

    _sub_z_mu: gp.MVar

    _std: gp.MVar
