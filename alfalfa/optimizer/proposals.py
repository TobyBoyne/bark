import numpy as np
import torch
from .optimizer_utils import \
    get_opt_core, add_gbm_to_opt_model, get_opt_core_copy, label_leaf_index, get_opt_sol
from ..utils.space import Space
from .gbm_model import GbmModel

import gurobipy

def propose(space: Space, opt_model: gurobipy.Model, gbm_model: GbmModel):
    next_x_area, next_val, curr_mean, curr_var = get_global_sol(space, opt_model, gbm_model)

    # add epsilon if input constr. exist
    # i.e. tree splits are rounded to the 5th decimal when adding them to the model,
    # and this may make optimization problems infeasible if the feasible region is very small
    # if self.model_core:
    #     self._add_epsilon_to_bnds(next_x_area)

    #     while True:
    #         try:
    #             next_center = self._get_leaf_min_center_dist(next_x_area)
    #             break
    #         except RuntimeError:
    #             self._add_epsilon_to_bnds(next_x_area)
    # else:
    next_center = _get_leaf_center(space, next_x_area)

    return next_center

def _get_leaf_center(space: Space, x_area):
    """returns the center of x_area"""
    next_x = []
    for idx in range(len(x_area)):
        if idx in space.cat_idx:
            # for cat vars
            xi = int(np.random.choice(list(x_area[idx]), size=1)[0])
        else:
            lb, ub = x_area[idx]

            if space.dims[idx].is_bin:
                # for bin vars
                if lb == 0 and ub == 1:
                    xi = int(np.random.randint(0, 2))
                elif lb <= 0.1:
                    xi = 0
                elif ub >= 0.9:
                    xi = 1
                else:
                    raise ValueError("problem with binary split, go to 'get_leaf_center'")

            elif idx in space.int_idx:
                # for int vars
                lb, ub = round(lb), round(ub)
                m = lb + (ub - lb) / 2
                xi = int(np.random.choice([int(m), round(m)], size=1)[0])

            else:
                # for conti vars
                xi = float(lb + (ub - lb) / 2)

        next_x.append(xi)
    return next_x


def get_global_sol(space: Space, opt_model: gurobipy.Model, gbm_model: GbmModel):
    # provides global solution to the optimization problem

    # build main model

    ## set solver parameters
    opt_model.Params.LogToConsole = 0
    opt_model.Params.Heuristics = 0.2
    opt_model.Params.TimeLimit = 100

    ## optimize opt_model to determine area to focus on
    opt_model.optimize()

    # get active leaf area
    label = '1st_obj'
    var_bnds = [d.bnds for d in space.dims]

    active_enc = \
        [(tree_id, leaf_enc) for tree_id, leaf_enc in label_leaf_index(opt_model, label)
        if round(opt_model._z_l[label, tree_id, leaf_enc].x) == 1.0]
    gbm_model.update_var_bounds(active_enc, var_bnds)

    # reading x_val
    next_x = get_opt_sol(space, opt_model)

    # extract variance and mean
    curr_var = opt_model._var.x
    curr_mean = sum([opt_model._mu_coeff[idx]*opt_model._sub_z_mu[idx].x
                        for idx in range(len(opt_model._mu_coeff))])

    return var_bnds, next_x, curr_mean, curr_var