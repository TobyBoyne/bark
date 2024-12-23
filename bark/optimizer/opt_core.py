from typing import TYPE_CHECKING

import gurobipy as gp
from beartype.typing import Optional
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
    NumericalInput,
)
from gurobipy import GRB, quicksum

from bark.bofire_utils.constraints import apply_constraint_to_model

if TYPE_CHECKING:
    from bark.optimizer.gbm_model import GbmModel


def get_opt_core(domain: Domain, env: Optional[gp.Env] = None) -> gp.Model:
    """Build the optimization core with input features"""
    model = gp.Model(env=env)
    model._cont_var_dict = {}
    model._cat_var_dict = {}

    for idx, feat in enumerate(domain.inputs.get()):
        var_name = feat.key

        if isinstance(feat, CategoricalInput):
            model._cat_var_dict[idx] = {}

            for i, cat in enumerate(feat.categories):
                model._cat_var_dict[idx][i] = model.addVar(
                    name=f"{var_name}_{cat}", vtype=GRB.BINARY
                )

            # constr vars need to add up to one
            model.addConstr(
                sum([model._cat_var_dict[idx][i] for i in range(len(feat.categories))])
                == 1
            )

        elif isinstance(feat, NumericalInput):
            if isinstance(feat, ContinuousInput):
                lb, ub = feat.bounds
                vtype = "C"
            elif isinstance(feat, DiscreteInput):
                lb, ub = feat.lower_bound, feat.upper_bound
                vtype = "B" if (lb, ub) == (0, 1) else "I"

            model._cont_var_dict[idx] = model.addVar(
                lb=lb, ub=ub, name=var_name, vtype=vtype
            )

    model._n_feat = len(model._cont_var_dict) + len(model._cat_var_dict)

    model.update()
    return model


def get_opt_core_copy(opt_core: gp.Model) -> gp.Model:
    """Create a copy of an optimization core."""
    new_opt_core = opt_core.copy()
    new_opt_core._n_feat = opt_core._n_feat

    # transfer var dicts
    new_opt_core._cont_var_dict = {}
    new_opt_core._cat_var_dict = {}

    ## transfer cont_var_dict
    for var in opt_core._cont_var_dict.keys():
        var_name = opt_core._cont_var_dict[var].VarName

        new_opt_core._cont_var_dict[var] = new_opt_core.getVarByName(var_name)

    ## transfer cat_var_dict
    for var in opt_core._cat_var_dict.keys():
        for cat in opt_core._cat_var_dict[var].keys():
            var_name = opt_core._cat_var_dict[var][cat].VarName

            if var not in new_opt_core._cat_var_dict.keys():
                new_opt_core._cat_var_dict[var] = {}

            new_opt_core._cat_var_dict[var][cat] = new_opt_core.getVarByName(var_name)

    return new_opt_core


def get_opt_core_from_domain(domain: Domain, env: Optional[gp.Env] = None) -> gp.Model:
    """Create an optimization model from a domain (including constraints)"""
    model_core = get_opt_core(domain, env=env)
    for constraint in domain.constraints:
        apply_constraint_to_model(constraint, model_core)

    model_core.Params.LogToConsole = 0
    model_core.Params.NonConvex = 2

    model_core.update()
    return model_core
    # add equality constraints to model core
    #     for func in self.eq_constr_funcs:
    #         model_core.addConstr(func(model_core._cont_var_dict) == 0.0)

    #     # add inequality constraints to model core
    #     for func in self.ineq_constr_funcs:
    #         model_core.addConstr(func(model_core._cont_var_dict) <= 0.0)

    #     # set solver parameter if function is nonconvex
    #     model_core.Params.LogToConsole = 0
    #     if self.is_nonconvex:
    #         model_core.Params.NonConvex = 2

    #     model_core.update()

    # return model_core


### GBT HANDLER
## gbt model helper functions


def label_leaf_index(model: gp.Model, label: str):
    for tree in range(model._num_trees(label)):
        for leaf in model._leaves(label, tree):
            yield (tree, leaf)


def tree_index(model: gp.Model):
    for label in model._gbm_set:
        for tree in range(model._num_trees(label)):
            yield (label, tree)


tree_index.dimen = 2


def leaf_index(model: gp.Model):
    for label, tree in tree_index(model):
        for leaf in model._leaves(label, tree):
            yield (label, tree, leaf)


leaf_index.dimen = 3


def misic_interval_index(model: gp.Model):
    for var in model._breakpoint_index:
        for j in range(len(model._breakpoints(var))):
            yield (var, j)


misic_interval_index.dimen = 2


def misic_split_index(model: gp.Model):
    gbm_models = model._gbm_models
    for label, tree in tree_index(model):
        for encoding in gbm_models[label].get_branch_encodings(tree):
            yield (label, tree, encoding)


misic_split_index.dimen = 3


def alt_interval_index(model: gp.Model):
    for var in model.breakpoint_index:
        for j in range(1, len(model.breakpoints[var]) + 1):
            yield (var, j)


alt_interval_index.dimen = 2


def add_gbm_to_opt_model(cat_idx: set[int], gbm_model_dict: dict, model: gp.Model):
    add_gbm_parameters(cat_idx, gbm_model_dict, model)
    add_gbm_variables(model)
    add_gbm_constraints(cat_idx, model)


def add_gbm_parameters(
    cat_idx: set[int], gbm_model_dict: dict[str, "GbmModel"], model: gp.Model
):
    model._gbm_models = gbm_model_dict

    model._gbm_set = set(gbm_model_dict.keys())
    model._num_trees = lambda label: gbm_model_dict[label].n_trees

    model._leaves = lambda label, tree: tuple(
        gbm_model_dict[label].get_leaf_encodings(tree)
    )

    model._leaf_weight = lambda label, tree, leaf: gbm_model_dict[
        label
    ].get_leaf_weight(tree, leaf)

    vbs = [v.get_var_break_points() for v in gbm_model_dict.values()]

    all_breakpoints = {}
    for i in range(model._n_feat):
        if i in cat_idx:
            continue
        s = set()
        for vb in vbs:
            if i in vb:
                s = s.union(set(vb[i]))

        if s:
            all_breakpoints[i] = sorted(s)

    model._breakpoint_index = list(all_breakpoints.keys())

    model._breakpoints = lambda i: all_breakpoints[i]

    model._leaf_vars = lambda label, tree, leaf: tuple(
        i for i in gbm_model_dict[label].get_participating_variables(tree, leaf)
    )


def add_gbm_variables(model: gp.Model):
    model._z_l = model.addVars(
        leaf_index(model), lb=0, ub=1, name="z_l", vtype=GRB.BINARY
    )

    model._y = model.addVars(misic_interval_index(model), name="y", vtype=GRB.BINARY)
    model.update()


def add_gbm_constraints(cat_idx, model):
    def single_leaf_rule(model_, label, tree):
        z_l, leaves = model_._z_l, model_._leaves
        return quicksum(z_l[label, tree, leaf] for leaf in leaves(label, tree)) == 1

    model.addConstrs(
        (single_leaf_rule(model, label, tree) for (label, tree) in tree_index(model)),
        name="single_leaf",
    )

    def left_split_r(model_, label, tree, split_enc):
        gbt = model_._gbm_models[label]
        split_var, split_val = gbt.get_branch_partition_pair(tree, split_enc)
        y_var = split_var

        if not isinstance(split_val, list):
            # for conti vars
            y_val = model_._breakpoints(y_var).index(split_val)
            return (
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_left_leaves(tree, split_enc)
                )
                <= model_._y[y_var, y_val]
            )
        else:
            # for cat vars
            return quicksum(
                model_._z_l[label, tree, leaf]
                for leaf in gbt.get_left_leaves(tree, split_enc)
            ) <= quicksum(model_._cat_var_dict[split_var][cat] for cat in split_val)

    def right_split_r(model_, label, tree, split_enc):
        gbt = model_._gbm_models[label]
        split_var, split_val = gbt.get_branch_partition_pair(tree, split_enc)
        y_var = split_var
        if not isinstance(split_val, list):
            # for conti vars
            y_val = model_._breakpoints(y_var).index(split_val)
            return (
                quicksum(
                    model_._z_l[label, tree, leaf]
                    for leaf in gbt.get_right_leaves(tree, split_enc)
                )
                <= 1 - model_._y[y_var, y_val]
            )
        else:
            # for cat vars
            return quicksum(
                model_._z_l[label, tree, leaf]
                for leaf in gbt.get_right_leaves(tree, split_enc)
            ) <= 1 - quicksum(model_._cat_var_dict[split_var][cat] for cat in split_val)

    def y_order_r(model_, i, j):
        if j == len(model_._breakpoints(i)):
            raise NotImplementedError("This constraint should be skipped")
        return model_._y[i, j] <= model_._y[i, j + 1]

    def cat_sums(model_, i):
        return (
            quicksum(
                model_._cat_var_dict[i][cat] for cat in model_._cat_var_dict[i].keys()
            )
            == 1
        )

    def var_lower_r(model_, i, j):
        lb = model_._cont_var_dict[i].lb
        j_bound = model_._breakpoints(i)[j]
        return model_._cont_var_dict[i] >= lb + (j_bound - lb) * (1 - model_._y[i, j])

    def var_upper_r(model_, i, j):
        ub = model_._cont_var_dict[i].ub
        j_bound = model_._breakpoints(i)[j]
        return model_._cont_var_dict[i] <= ub + (j_bound - ub) * (model_._y[i, j])

    model.addConstrs(
        (
            left_split_r(model, label, tree, encoding)
            for (label, tree, encoding) in misic_split_index(model)
        ),
        name="left_split",
    )

    model.addConstrs(
        (
            right_split_r(model, label, tree, encoding)
            for (label, tree, encoding) in misic_split_index(model)
        ),
        name="right_split",
    )

    # for conti vars
    model.addConstrs(
        (
            y_order_r(model, var, j)
            for (var, j) in misic_interval_index(model)
            if j != len(model._breakpoints(var)) - 1
        ),
        name="y_order",
    )

    # for cat vars
    model.addConstrs((cat_sums(model, var) for var in cat_idx), name="cat_sums")

    model.addConstrs(
        (var_lower_r(model, var, j) for (var, j) in misic_interval_index(model)),
        name="var_lower",
    )

    model.addConstrs(
        (var_upper_r(model, var, j) for (var, j) in misic_interval_index(model)),
        name="var_upper",
    )
