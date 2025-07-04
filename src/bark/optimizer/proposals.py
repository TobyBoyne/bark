import logging

import gurobipy as gp
import numpy as np
from beartype.typing import Optional
from bofire.data_models.domain.api import Domain, Inputs
from bofire.data_models.features.api import DiscreteInput
from gurobipy import GRB

from bofire_mixed.domain import get_cat_idx_from_domain, get_feature_bounds

from .opt_core import (
    get_opt_core_copy,
    label_leaf_index,
)

logging.getLogger("gurobipy").setLevel(logging.ERROR)


def get_opt_sol(input_feats: Inputs, cat_idx: set[int], opt_model: gp.Model):
    # get optimal solution from gurobi model
    next_x = []
    for idx, feat in enumerate(input_feats):
        x_val = None
        try:
            if idx in cat_idx:
                # check which category is active
                for cat_i in range(len(feat.categories)):
                    if opt_model._cat_var_dict[idx][cat_i].x > 0.5:
                        x_val = cat_i
            else:
                x_val = opt_model._cont_var_dict[idx].x

        except AttributeError:
            raise ValueError(
                f"Gurobi was unable to converge; ended with status code {opt_model.Status} (failed on{feat.key}). See https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html#secstatuscodes for more information."
            )

        next_x.append(x_val)
    return next_x


def propose(
    domain: Domain,
    opt_model: gp.Model,
    model_core: Optional[gp.Model],
):
    cat_idx = get_cat_idx_from_domain(domain)
    input_features = domain.inputs.get()

    next_x_area, next_val = _get_global_sol(input_features, cat_idx, opt_model)

    # add epsilon if input constr. exist
    # i.e. tree splits are rounded to the 5th decimal when adding them to the model,
    # and this may make optimization problems infeasible if the feasible region is very small
    if model_core is not None:
        _add_epsilon_to_bnds(next_x_area, input_features, cat_idx)

        while True:
            try:
                next_center = _get_leaf_min_center_dist(
                    next_x_area, input_features, cat_idx, model_core
                )
                break
            except ValueError:
                _add_epsilon_to_bnds(next_x_area, input_features, cat_idx)
    else:
        next_center = _get_leaf_center(next_x_area, input_features, cat_idx)

    return next_center


def _get_global_sol(
    input_feats: Inputs,
    cat_idx: set[int],
    opt_model: gp.Model,
    time_limit: int = 100,
):
    # provides global solution to the optimization problem

    # build main model

    ## set solver parameters
    opt_model.Params.LogToConsole = 0
    opt_model.Params.Heuristics = 0.2
    opt_model.Params.TimeLimit = time_limit
    opt_model.Params.MIPGap = 0.10
    opt_model.Params.LogFile = "gurobi.log"
    opt_model.Params.MIPFocus = 0
    opt_model.Params.NonConvex = 0

    ## optimize opt_model to determine area to focus on
    opt_model.optimize()

    var_bnds = [get_feature_bounds(feat, encoding="ordinal") for feat in input_feats]

    # get active leaf area
    errors = []
    for label, gbm_model in opt_model._gbm_models.items():
        present_solns = [
            (tree_id, leaf_enc)
            for tree_id, leaf_enc in label_leaf_index(opt_model, label)
            if hasattr(opt_model._z_l[label, tree_id, leaf_enc], "x")
        ]
        if not present_solns:
            errors.append(label)
        active_enc = [
            (tree_id, leaf_enc)
            for tree_id, leaf_enc in present_solns
            if round(opt_model._z_l[label, tree_id, leaf_enc].x) == 1.0
        ]
        gbm_model.update_var_bounds(active_enc, var_bnds)
    if errors:
        # Would use exceptiongroups but not supported by python 3.10
        raise ValueError(f"No active solutions found for labels {errors}")
    # reading x_val
    next_x = get_opt_sol(input_feats, cat_idx, opt_model)

    # extract variance and mean
    # curr_var = opt_model._var.x
    # curr_mean = sum(
    #     [
    #         opt_model._mu_coeff[idx] * opt_model._sub_z_mu[idx].x
    #         for idx in range(len(opt_model._mu_coeff))
    #     ]
    # )

    return var_bnds, next_x


def _get_leaf_center(x_area, input_feats: Inputs, cat_idx: set[int]):
    """returns the center of x_area"""
    next_x = []
    for idx, feat in enumerate(input_feats):
        if idx in cat_idx:
            # for cat vars
            xi = int(np.random.choice(list(x_area[idx]), size=1)[0])
        else:
            lb, ub = x_area[idx]
            xi = lb + (ub - lb) / 2
            if isinstance(feat, DiscreteInput):
                xi_floor = int(np.floor(xi))
                xi_remainder = xi - xi_floor
                xi = xi_floor + np.random.binomial(1, xi_remainder, size=1).item()

        next_x.append(xi)
    return next_x


def _get_leaf_min_center_dist(
    x_area, input_feats: Inputs, cat_idx: set[int], model_core: gp.Model
):
    """returns the feasible point closest to the x_area center"""
    # build opt_model core

    opt_model = get_opt_core_copy(model_core)

    # define alpha as the distance to closest data point
    opt_model._alpha = opt_model.addVar(lb=0.0, ub=GRB.INFINITY, name="alpha")

    # update bounds for all variables
    for idx, feat in enumerate(input_feats):
        if idx in cat_idx:
            # add constr for cat vars
            for cat_i, cat in enumerate(feat.categories):
                # cat is fixed to what is valid with respect to x_area[idx]
                if cat_i not in x_area[idx]:
                    opt_model.addConstr(opt_model._cat_var_dict[idx][cat_i] == 0)
        else:
            lb, ub = x_area[idx]
            opt_model.addConstr(opt_model._cont_var_dict[idx] <= ub)
            opt_model.addConstr(opt_model._cont_var_dict[idx] >= lb)

    # add constraints for every data point
    x_center = _get_leaf_center(x_area, input_feats, cat_idx)

    for x in [x_center]:
        expr = []

        # add dist for all dimensions
        for idx, feat in enumerate(input_feats):
            if idx in cat_idx:
                # add constr for cat vars
                for cat_i, cat in enumerate(feat.categories):
                    # distance increases by one if cat is different from x[idx]
                    if cat_i != x[idx]:
                        expr.append(opt_model._cat_var_dict[idx][cat_i])

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

    return get_opt_sol(input_feats, cat_idx, opt_model)


def _add_epsilon_to_bnds(x_area, input_feats: Inputs, cat_idx: set[int]):
    # adds a 1e-5 error to the bounds of area
    eps = 1e-5
    for idx, feat in enumerate(input_feats):
        if idx not in cat_idx:
            lb, ub = x_area[idx]
            feat_lb, feat_ub = get_feature_bounds(feat)
            new_lb = max(lb - eps, feat_lb)
            new_ub = min(ub + eps, feat_ub)
            x_area[idx] = (new_lb, new_ub)
