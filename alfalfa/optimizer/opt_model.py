"""From Leaf-GP"""

from typing import Optional

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB, MVar
from scipy.linalg import cho_factor, cho_solve

from ..tree_kernels import AlfalfaGP, AlfalfaMOGP
from ..utils.space import Space
from .gbm_model import GbmModel
from .optimizer_utils import add_gbm_to_opt_model, get_opt_core


def build_opt_model(
    space: Space,
    gbm_model: GbmModel,
    tree_gp: AlfalfaGP,
    kappa: float,
    model_core: Optional[gp.Model],
):
    # build opt_model core

    # check if there's already a model core with extra constraints
    if model_core is None:
        opt_model = get_opt_core(space)
    else:
        # copy model core in case there are constr given already
        opt_model = get_opt_core_copy(model_core)
    opt_model = get_opt_core(space)

    # build tree model
    gbm_model_dict = {"1st_obj": gbm_model}
    add_gbm_to_opt_model(space, gbm_model_dict, opt_model)

    if isinstance(tree_gp, AlfalfaMOGP):
        # multi-fidelity case - evaluate for highest fidelity
        # get tree_gp hyperparameters
        kernel_var = (
            tree_gp.task_covar_module._eval_covar_matrix()[0, 0].detach().numpy()
        )
        noise_var = tree_gp.likelihood.noise[0].detach().numpy()

        # get tree_gp matrices
        (train_x_all, train_i) = tree_gp.train_inputs
        # only optimise on highest fidelity
        target_f = (train_i == 0).flatten()
        assert target_f.any(), "No data for highest fidelity"
        train_x = train_x_all[..., target_f, :]
        Kmm = tree_gp.covar_module(train_x).numpy()
        k_diag = np.diagonal(Kmm)
        s_diag = tree_gp.likelihood._shaped_noise_covar(
            k_diag.shape, [torch.zeros((k_diag.shape[0], 1))]
        ).numpy()
        s_diag = s_diag.squeeze(-3)

        y_vals = tree_gp.train_targets[target_f].numpy()

    else:
        # get tree_gp hyperparameters
        kernel_var = tree_gp.covar_module.outputscale.detach().numpy()
        noise_var = tree_gp.likelihood.noise.detach().numpy()

        # get tree_gp matrices
        (train_x,) = tree_gp.train_inputs
        Kmm = tree_gp.covar_module(train_x).numpy()
        k_diag = np.diagonal(Kmm)
        s_diag = tree_gp.likelihood._shaped_noise_covar(k_diag.shape).numpy()

        y_vals = tree_gp.train_targets.numpy()

    ks = Kmm + s_diag

    # invert gram matrix

    id_ks = np.eye(ks.shape[0])
    inv_ks = cho_solve(cho_factor(ks, lower=True), id_ks)

    # add tree_gp logic to opt_model

    act_leave_vars = gbm_model.get_active_leaf_vars(
        train_x.numpy(), opt_model, "1st_obj"
    )

    sub_k = opt_model.addVars(
        range(len(act_leave_vars)), lb=0, ub=1, name="sub_k", vtype="C"
    )

    opt_model.addConstrs(
        (sub_k[idx] == act_leave_vars[idx] for idx in range(len(act_leave_vars))),
        name="sub_k_constr",
    )

    ## add quadratic constraints
    # \sigma <= K_xx - K_xX @ K_XX^-1 @ X_xX^T
    opt_model._var = opt_model.addVar(lb=0, ub=GRB.INFINITY, name="var", vtype="C")

    opt_model._sub_k_var = MVar(sub_k.values() + [opt_model._var])

    quadr_term = -(kernel_var**2) * inv_ks
    const_term = kernel_var + noise_var

    zeros = np.zeros((train_x.shape[0], 1))
    quadr_constr = np.block([[quadr_term, zeros], [zeros.T, -1.0]])

    opt_model.addMQConstr(
        quadr_constr,
        None,
        sense=">",
        rhs=-const_term,
        xQ_L=opt_model._sub_k_var,
        xQ_R=opt_model._sub_k_var,
    )

    ## add linear objective
    opt_model._sub_z_obj = MVar(sub_k.values() + [opt_model._var])

    # why multiply by kernel var here?
    lin_term = kernel_var * (inv_ks @ y_vals)

    lin_obj = np.concatenate((lin_term, [-kappa]))

    opt_model.setMObjective(
        None, lin_obj, 0, xc=opt_model._sub_z_obj, sense=GRB.MINIMIZE
    )

    ## add mu variable
    opt_model._sub_z_mu = MVar(sub_k.values())
    opt_model._mu_coeff = lin_term

    return opt_model


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
