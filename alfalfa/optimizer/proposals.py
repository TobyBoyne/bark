import gurobipy as gp
import numpy as np
from beartype.typing import Optional
from gurobipy import GRB

from ..utils.space import Space
from .gbm_model import GbmModel
from .optimizer_utils import (
    get_opt_core_copy,
    get_opt_sol,
    label_leaf_index,
)


def propose(
    space: Space,
    opt_model: gp.Model,
    gbm_model: GbmModel,
    model_core: Optional[gp.Model] = None,
):
    next_x_area, next_val, curr_mean, curr_var = get_global_sol(
        space, opt_model, gbm_model
    )

    # add epsilon if input constr. exist
    # i.e. tree splits are rounded to the 5th decimal when adding them to the model,
    # and this may make optimization problems infeasible if the feasible region is very small
    if model_core is not None:
        _add_epsilon_to_bnds(next_x_area, space)

        while True:
            try:
                next_center = _get_leaf_min_center_dist(next_x_area, space, model_core)
                break
            except ValueError:
                _add_epsilon_to_bnds(next_x_area, space)
    else:
        next_center = _get_leaf_center(next_x_area, space)

    return next_center


def get_global_sol(space: Space, opt_model: gp.Model, gbm_model: GbmModel):
    # provides global solution to the optimization problem

    # build main model

    ## set solver parameters
    opt_model.Params.LogToConsole = 0
    opt_model.Params.Heuristics = 0.2
    opt_model.Params.TimeLimit = 100

    ## optimize opt_model to determine area to focus on
    opt_model.optimize()

    # get active leaf area
    label = "1st_obj"
    var_bnds = [d.bnds for d in space.dims]

    active_enc = [
        (tree_id, leaf_enc)
        for tree_id, leaf_enc in label_leaf_index(opt_model, label)
        if round(opt_model._z_l[label, tree_id, leaf_enc].x) == 1.0
    ]
    gbm_model.update_var_bounds(active_enc, var_bnds)

    # reading x_val
    next_x = get_opt_sol(space, opt_model)

    # extract variance and mean
    curr_var = opt_model._var.x
    curr_mean = sum(
        [
            opt_model._mu_coeff[idx] * opt_model._sub_z_mu[idx].x
            for idx in range(len(opt_model._mu_coeff))
        ]
    )

    return var_bnds, next_x, curr_mean, curr_var


def _get_leaf_center(x_area, space: Space):
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
                    raise ValueError(
                        "problem with binary split, go to 'get_leaf_center'"
                    )

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


def _get_leaf_min_center_dist(x_area, space: Space, model_core: gp.Model):
    """returns the feasible point closest to the x_area center"""
    # build opt_model core

    opt_model = get_opt_core_copy(model_core)

    # define alpha as the distance to closest data point
    opt_model._alpha = opt_model.addVar(lb=0.0, ub=GRB.INFINITY, name="alpha")

    # update bounds for all variables
    for idx in range(len(space.dims)):
        if idx in space.cat_idx:
            # add constr for cat vars
            cat_set = set(space.dims[idx].bnds)

            for cat in cat_set:
                # cat is fixed to what is valid with respect to x_area[idx]
                if cat not in x_area[idx]:
                    opt_model.addConstr(opt_model._cat_var_dict[idx][cat] == 0)
        else:
            lb, ub = x_area[idx]
            opt_model.addConstr(opt_model._cont_var_dict[idx] <= ub)
            opt_model.addConstr(opt_model._cont_var_dict[idx] >= lb)

    # add constraints for every data point
    x_center = _get_leaf_center(x_area, space)

    for x in [x_center]:
        expr = []

        # add dist for all dimensions
        for idx in range(len(space.dims)):
            if idx in space.cat_idx:
                # add constr for cat vars
                cat_set = set(space.dims[idx].bnds)

                for cat in cat_set:
                    # distance increases by one if cat is different from x[idx]
                    if cat != x[idx]:
                        expr.append(opt_model._cat_var_dict[idx][cat])

            else:
                # add constr for conti and int vars
                expr.append((x[idx] - opt_model._cont_var_dict[idx]) ** 2)

        # add dist constraints to model
        opt_model.addConstr(opt_model._alpha >= sum(expr))

    # set optimization parameters
    opt_model.Params.LogToConsole = 0
    opt_model.Params.NonConvex = 2
    opt_model.setObjective(expr=opt_model._alpha)
    opt_model.optimize()

    return get_opt_sol(space, opt_model)


def _add_epsilon_to_bnds(x_area, space: Space):
    # adds a 1e-5 error to the bounds of area
    eps = 1e-5
    for idx in range(len(space.dims)):
        if idx not in space.cat_idx:
            lb, ub = x_area[idx]
            new_lb = max(lb - eps, space.dims[idx].bnds[0])
            new_ub = min(ub + eps, space.dims[idx].bnds[1])
            x_area[idx] = (new_lb, new_ub)
