import gurobipy as gp
import numpy as np
from beartype.typing import Optional
from gurobipy import GRB, quicksum

from ..utils.space import Space


def conv2list(x):
    temp_x = []
    for xi in x:
        if isinstance(xi, np.int_):
            temp_x.append(int(xi))
        elif isinstance(xi, np.str_):
            temp_x.append(str(xi))
        else:
            temp_x.append(xi)
    return temp_x


def get_opt_sol(space: Space, opt_model: gp.Model):
    # get optimal solution from gurobi model
    next_x = []
    for idx in range(len(space.dims)):
        x_val = None
        if idx in space.cat_idx:
            # check which category is active
            cat_set = set(space.dims[idx].bnds)
            for cat in cat_set:
                if opt_model._cat_var_dict[idx][cat].x > 0.5:
                    x_val = cat
        else:
            try:
                x_val = opt_model._cont_var_dict[idx].x
            except AttributeError:
                pass

        if x_val is None:
            raise ValueError(
                f"'get_opt_sol' wasn't able to extract solution for feature {idx}"
            )

        next_x.append(x_val)
    return next_x


def get_opt_core(space: Space, opt_core: Optional[gp.Model] = None):
    """creates the base optimization model"""
    if opt_core is None:
        model = gp.Model()
        model._cont_var_dict = {}
        model._cat_var_dict = {}
    else:
        return opt_core

    for idx, d in enumerate(space.dims):
        var_name = "_".join(["x", str(idx)])

        if idx in space.cont_idx or idx in space.int_idx:
            lb = d.bnds[0]
            ub = d.bnds[1]

            if d.var_type == "int":
                if d.is_bin:
                    # define binary vars
                    model._cont_var_dict[idx] = model.addVar(name=var_name, vtype="B")
                else:
                    # define integer vars
                    model._cont_var_dict[idx] = model.addVar(
                        lb=lb, ub=ub, name=var_name, vtype="I"
                    )
            else:
                # define continuous vars
                model._cont_var_dict[idx] = model.addVar(
                    lb=lb, ub=ub, name=var_name, vtype="C"
                )

        elif idx in space.cat_idx:
            # define categorical vars
            model._cat_var_dict[idx] = {}

            for cat in d.bnds:
                model._cat_var_dict[idx][cat] = model.addVar(
                    name=f"{var_name}_{cat}", vtype=GRB.BINARY
                )

            # constr vars need to add up to one
            model.addConstr(sum([model._cat_var_dict[idx][cat] for cat in d.bnds]) == 1)

    model._n_feat = len(model._cont_var_dict) + len(model._cat_var_dict)

    model.update()
    return model


def get_opt_core_copy(opt_core: gp.Model):
    """creates the copy of an optimization model"""
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


### GBT HANDLER
## gbt model helper functions


def label_leaf_index(model, label):
    for tree in range(model._num_trees(label)):
        for leaf in model._leaves(label, tree):
            yield (tree, leaf)


def tree_index(model):
    for label in model._gbm_set:
        for tree in range(model._num_trees(label)):
            yield (label, tree)


tree_index.dimen = 2


def leaf_index(model):
    for label, tree in tree_index(model):
        for leaf in model._leaves(label, tree):
            yield (label, tree, leaf)


leaf_index.dimen = 3


def misic_interval_index(model):
    for var in model._breakpoint_index:
        for j in range(len(model._breakpoints(var))):
            yield (var, j)


misic_interval_index.dimen = 2


def misic_split_index(model):
    gbm_models = model._gbm_models
    for label, tree in tree_index(model):
        for encoding in gbm_models[label].get_branch_encodings(tree):
            yield (label, tree, encoding)


misic_split_index.dimen = 3


def alt_interval_index(model):
    for var in model.breakpoint_index:
        for j in range(1, len(model.breakpoints[var]) + 1):
            yield (var, j)


alt_interval_index.dimen = 2


def add_gbm_to_opt_model(space: Space, gbm_model_dict: dict, model: gp.Model):
    add_gbm_parameters(space.cat_idx, gbm_model_dict, model)
    add_gbm_variables(model)
    add_gbm_constraints(space.cat_idx, model)


def add_gbm_parameters(cat_idx, gbm_model_dict, model):
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
        else:
            s = set()
            for vb in vbs:
                try:
                    s = s.union(set(vb[i]))
                except KeyError:
                    pass
            if s:
                all_breakpoints[i] = sorted(s)

    model._breakpoint_index = list(all_breakpoints.keys())

    model._breakpoints = lambda i: all_breakpoints[i]

    model._leaf_vars = lambda label, tree, leaf: tuple(
        i for i in gbm_model_dict[label].get_participating_variables(tree, leaf)
    )


def add_gbm_variables(model):
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
