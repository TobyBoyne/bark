from typing import cast

import gurobipy as gp
from beartype.typing import Optional
from gurobipy import GRB, quicksum

from bark import types
from bark.enums import FeatureTypeEnum
from bark.optimizer.mip_model import TreesMIPModel
from bark.types.optimizer import GurobiOptimizerModel
from bark.utils.bit_operations import next_power_of_2_exponent


def get_opt_core_copy(opt_core: GurobiOptimizerModel) -> GurobiOptimizerModel:
    """Create a copy of an optimization core."""
    new_opt_core = cast(GurobiOptimizerModel, opt_core.copy())
    new_opt_core._n_feat = opt_core._n_feat

    # transfer var dicts
    new_opt_core._cont_var_dict = {}
    new_opt_core._cat_var_dict = {}

    ## transfer cont_var_dict
    for var in opt_core._cont_var_dict.keys():
        var_name = opt_core._cont_var_dict[var].VarName

        new_var = new_opt_core.getVarByName(var_name)
        assert new_var is not None
        new_opt_core._cont_var_dict[var] = new_var

    ## transfer cat_var_dict
    for var in opt_core._cat_var_dict.keys():
        for cat in opt_core._cat_var_dict[var].keys():
            var_name = opt_core._cat_var_dict[var][cat].VarName

            if var not in new_opt_core._cat_var_dict.keys():
                new_opt_core._cat_var_dict[var] = {}

            new_var = new_opt_core.getVarByName(var_name)
            assert new_var is not None
            new_opt_core._cat_var_dict[var][cat] = new_var

    return new_opt_core


def get_opt_core_from_bark_data(
    data: types.Data, env: Optional[gp.Env] = None
) -> GurobiOptimizerModel:
    """Create an optimization model from a domain (including constraints)"""
    model = cast(GurobiOptimizerModel, gp.Model(env=env))
    model._cont_var_dict = {}
    model._cat_var_dict = {}

    for idx, feat_type in enumerate(data.feat_types):
        var_name = f"input_{idx}"

        if feat_type == FeatureTypeEnum.Cat:
            model._cat_var_dict[idx] = {}
            num_categories = next_power_of_2_exponent(data.bounds[1, idx])

            for i in range(num_categories):
                model._cat_var_dict[idx][i] = model.addVar(
                    name=f"{var_name}_{i}", vtype=GRB.BINARY
                )

            # constr vars need to add up to one
            model.addConstr(
                quicksum([model._cat_var_dict[idx][i] for i in range(num_categories)])
                == 1
            )

        else:
            lb, ub = data.bounds[:, idx]
            if feat_type == FeatureTypeEnum.Cont:
                vtype = "C"
            else:
                vtype = "B" if (lb, ub) == (0, 1) else "I"

            model._cont_var_dict[idx] = model.addVar(
                lb=lb, ub=ub, name=var_name, vtype=vtype
            )

    model._n_feat = len(model._cont_var_dict) + len(model._cat_var_dict)

    model.update()
    return model


### GBT HANDLER
## gbt model helper functions


def label_leaf_index(model: GurobiOptimizerModel, label: str):
    for tree in range(model._num_trees(label)):
        for leaf in model._leaves(label, tree):
            yield (tree, leaf)


def tree_index(model: GurobiOptimizerModel):
    for label in model._trees_set:
        for tree in range(model._num_trees(label)):
            yield (label, tree)


def leaf_index(model: GurobiOptimizerModel):
    for label, tree in tree_index(model):
        for leaf in model._leaves(label, tree):
            yield (label, tree, leaf)


def misic_interval_index(model: GurobiOptimizerModel):
    for var in model._breakpoint_index:
        for j in range(len(model._breakpoints(var))):
            yield (var, j)


def misic_split_index(model: GurobiOptimizerModel):
    tree_models = model._tree_models
    for label, tree in tree_index(model):
        for encoding in tree_models[label].get_branch_encodings(tree):
            yield (label, tree, encoding)


def add_trees_to_opt_model(
    cat_idx: set[int],
    trees_mip_model_dict: dict[str, TreesMIPModel],
    model: GurobiOptimizerModel,
):
    add_tree_parameters(cat_idx, trees_mip_model_dict, model)
    add_tree_variables(model)
    add_tree_constraints(cat_idx, model)


def add_tree_parameters(
    cat_idx: set[int],
    trees_mip_model_dict: dict[str, TreesMIPModel],
    model: GurobiOptimizerModel,
):
    model._tree_models = trees_mip_model_dict

    model._trees_set = set(trees_mip_model_dict.keys())
    model._num_trees = lambda label: trees_mip_model_dict[label].n_trees

    model._leaves = lambda label, tree: tuple(
        trees_mip_model_dict[label].get_leaf_encodings(tree)
    )

    # model._leaf_weight = lambda label, tree, leaf: gbm_model_dict[
    #     label
    # ].get_leaf_weight(tree, leaf)

    vbs = [v.get_var_break_points() for v in trees_mip_model_dict.values()]

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

    # model._leaf_vars = lambda label, tree, leaf: tuple(
    #     i for i in gbm_model_dict[label].get_participating_variables(tree, leaf)
    # )


def add_tree_variables(model: GurobiOptimizerModel):
    model._z_l = model.addVars(
        leaf_index(model), lb=0, ub=1, name="z_l", vtype=GRB.BINARY
    )

    model._y = model.addVars(misic_interval_index(model), name="y", vtype=GRB.BINARY)
    model.update()


def add_tree_constraints(cat_idx, model: GurobiOptimizerModel):
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
