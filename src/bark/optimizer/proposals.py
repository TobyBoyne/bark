import logging

import numpy as np
from gurobipy import GRB
from jaxtyping import Array, Float

from bark import types
from bark.enums import FeatureTypeEnum
from bark.types.optimizer import GurobiOptimizerModel

from .opt_core import (
    get_opt_core_copy,
    label_leaf_index,
)

logging.getLogger("gurobipy").setLevel(logging.ERROR)


def _features_as_list(
    features: types.Features,
) -> list[tuple[float, float] | list[int]]:
    def aslist(bound: Float[Array, " 2"], feat_type: FeatureTypeEnum):
        if feat_type == FeatureTypeEnum.Cat:
            return list(range(bound[1]))
        else:
            return (float(bound[0]), float(bound[1]))

    ordinal_bounds = features.ordinal_bounds
    return [aslist(b, f) for b, f in zip(ordinal_bounds.T, features.feat_types)]


def get_opt_sol(feat_types: types.FeatTypesT, opt_model: GurobiOptimizerModel):
    # get optimal solution from gurobi model
    next_x = []
    for idx, feat_type in enumerate(feat_types):
        x_val = None
        try:
            if feat_type == FeatureTypeEnum.Cat:
                # check which category is active
                category_dict = opt_model._cat_var_dict[idx]
                for cat_i, var in category_dict.items():
                    if var.X > 0.5:
                        x_val = cat_i
            else:
                x_val = opt_model._cont_var_dict[idx].X

        except AttributeError:
            raise ValueError(
                f"Gurobi was unable to converge; ended with status code {opt_model.Status} (failed on {idx}). See https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html#secstatuscodes for more information."
            )

        next_x.append(x_val)
    return next_x


def propose(
    features: types.Features,
    opt_model: GurobiOptimizerModel,
    model_core: GurobiOptimizerModel | None = None,
):
    next_x_area, next_val = _get_global_sol(features, opt_model)

    # add epsilon if input constr. exist
    # i.e. tree splits are rounded to the 5th decimal when adding them to the model,
    # and this may make optimization problems infeasible if the feasible region is very small
    if model_core is not None:
        _add_epsilon_to_bnds(next_x_area, features.feat_types)

        while True:
            try:
                next_center = _get_leaf_min_center_dist(
                    next_x_area, features.feat_types, model_core
                )
                break
            except ValueError:
                _add_epsilon_to_bnds(next_x_area, features.feat_types)
    else:
        next_center = _get_leaf_center(next_x_area, features.feat_types)

    return next_center


def _get_global_sol(
    features: types.Features,
    opt_model: GurobiOptimizerModel,
    time_limit: int = 100,
):
    # provides global solution to the optimization problem

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

    var_bnds = _features_as_list(features)

    # get active leaf area
    errors = []
    for label, tree_model in opt_model._tree_models.items():
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
            if round(opt_model._z_l[label, tree_id, leaf_enc].X) == 1.0
        ]
        tree_model.update_var_bounds_inplace(active_enc, var_bnds)
    if errors:
        # Would use exceptiongroups but not supported by python 3.10
        raise ValueError(f"No active solutions found for labels {errors}")
    # reading x_val
    next_x = get_opt_sol(features.feat_types, opt_model)

    return var_bnds, next_x


def _get_leaf_center(x_area, feat_types: types.FeatTypesT):
    """returns the center of x_area"""
    next_x = []
    for idx, feat_type in enumerate(feat_types):
        if feat_type == FeatureTypeEnum.Cat:
            # for cat vars
            xi = int(np.random.choice(list(x_area[idx]), size=1)[0])
        else:
            lb, ub = x_area[idx]
            xi = lb + (ub - lb) / 2
            if feat_type == FeatureTypeEnum.Int:
                xi_floor = int(np.floor(xi))
                xi_remainder = xi - xi_floor
                xi = xi_floor + np.random.binomial(1, xi_remainder, size=1).item()

        next_x.append(xi)
    return next_x


def _get_leaf_min_center_dist(
    x_area, feat_types: types.FeatTypesT, model_core: GurobiOptimizerModel
):
    """returns the feasible point closest to the x_area center"""
    # build opt_model core

    opt_model = get_opt_core_copy(model_core)

    # define alpha as the distance to closest data point
    opt_model._alpha = opt_model.addVar(lb=0.0, ub=GRB.INFINITY, name="alpha")

    # update bounds for all variables
    for idx, feat_type in enumerate(feat_types):
        if feat_type == FeatureTypeEnum.Cat:
            # add constr for cat vars
            category_dict = opt_model._cat_var_dict[idx]
            for cat_i in category_dict.keys():
                # cat is fixed to what is valid with respect to x_area[idx]
                if cat_i not in x_area[idx]:
                    opt_model.addConstr(opt_model._cat_var_dict[idx][cat_i] == 0)
        else:
            lb, ub = x_area[idx]
            opt_model.addConstr(opt_model._cont_var_dict[idx] <= ub)
            opt_model.addConstr(opt_model._cont_var_dict[idx] >= lb)

    # add constraints for every data point
    x_center = _get_leaf_center(x_area, feat_types)

    for x in [x_center]:
        expr = []

        # add dist for all dimensions
        for idx, feat_type in enumerate(feat_types):
            if feat_type == FeatureTypeEnum.Cat:
                category_dict = opt_model._cat_var_dict[idx]
                # add constr for cat vars
                for cat_i in category_dict.keys():
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

    return get_opt_sol(feat_types, opt_model)


def _add_epsilon_to_bnds(x_area, feat_types: types.FeatTypesT):
    # adds a 1e-5 error to the bounds of area
    eps = 1e-5
    for idx, feat_type in enumerate(feat_types):
        if feat_type != FeatureTypeEnum.Cat:
            lb, ub = x_area[idx]
            # This should really take the feature bounds into account
            # feat_lb, feat_ub = get_feature_bounds(feat)
            new_lb = lb - eps
            new_ub = ub + eps
            x_area[idx] = (new_lb, new_ub)
