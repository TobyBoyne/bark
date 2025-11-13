"""From Leaf-GP"""

import gurobipy as gp
import jax
import jax.numpy as jnp
import numpy as np
import torch
from beartype.typing import Optional
from bofire.data_models.domain.api import Domain
from gurobipy import GRB, MVar
from scipy.linalg import cho_factor, cho_solve

from bark import forest, types
from bark.enums import FeatureTypeEnum
from bark.tree_kernels.tree_gps import LeafGP
from bofire_mixed.domain import get_cat_idx_from_domain

from .gbm_model import GbmModel
from .opt_core import add_gbm_to_opt_model, get_opt_core, get_opt_core_copy


def build_opt_model_from_forest(
    bark_model: types.BARKModel,
    data: types.Data,
    kappa: float,
    model_core: gp.Model,
):
    opt_model = get_opt_core_copy(model_core)
    train_X, train_Y = data.train_X, data.train_Y
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()

    # unpack samples
    bark_model = bark_model.get_flattened_samples()

    num_samples = bark_model.batch_shape[0]
    num_data = train_X.shape[0]

    # build tree model
    gbm_model_dict: dict[str, GbmModel] = {}
    for sample_idx in range(num_samples):
        trees = jax.tree_util.tree_map(lambda t: t[sample_idx], bark_model.trees)
        gbm_model_dict[f"tree_sample_{sample_idx}"] = GbmModel(trees, data.feat_types)

    cat_idx = {i for i, f in data.feat_types if f == FeatureTypeEnum.Cat}
    add_gbm_to_opt_model(cat_idx, gbm_model_dict, opt_model)
    K_XX = jax.vmap(forest.forest_gram_matrix_no_null, in_axes=(None, 0, None))(
        train_X, bark_model.trees, data.feat_types
    )

    K_XX_s = K_XX + (1e-6 + bark_model.noise_var[:, None, None]) * np.eye(num_data)
    # cholesky decomposition doesn't support batching
    K_inv = jnp.linalg.inv(K_XX_s)

    num_sub_k = num_samples * num_data
    sub_k = opt_model.addVars(range(num_sub_k), lb=0, ub=1, name="sub_k", vtype="C")
    for i, (gbm_name, gbm_model) in enumerate(gbm_model_dict.items()):
        # create active leaf variables
        act_leave_vars = gbm_model.get_active_leaf_vars(train_X, opt_model, gbm_name)

        opt_model.addConstrs(
            (
                sub_k[idx + i * len(act_leave_vars)] == act_leave_vars[idx]
                for idx in range(len(act_leave_vars))
            ),
            name=f"sub_k_constr_{gbm_name}",
        )

    ## add quadratic constraints
    # \sigma <= K_xx - K_xX @ K_XX^-1 @ X_xX^T

    opt_model._std = opt_model.addVars(
        range(num_samples), lb=0, ub=GRB.INFINITY, name="std", vtype="C"
    )

    # pre- and post-multiply by scale
    quadr_term = -K_inv
    const_term = jnp.ones_like(bark_model.noise_var)
    zeros = np.zeros((train_X.shape[0], 1))

    for i in range(num_samples):
        quadr_constr = np.block([[quadr_term[i], zeros], [zeros.T, -1.0]])
        sub_k_sample = [sub_k[j] for j in range(i * num_data, (i + 1) * num_data)]
        sub_k_std = MVar.fromlist(sub_k_sample + [opt_model._std[i]])
        opt_model.addMQConstr(  # type: ignore
            quadr_constr,
            None,
            sense=">",
            rhs=-const_term[i],
            xQ_L=sub_k_std,
            xQ_R=sub_k_std,
        )

    ## add linear objective
    lin_term = K_inv @ train_Y[None, :, :]
    lin_term = lin_term.squeeze(-1)

    obj = 0
    for i in range(num_samples):
        sub_z_sample = [sub_k[j] for j in range(i * num_data, (i + 1) * num_data)]
        sub_z_obj = MVar.fromlist(sub_z_sample + [opt_model._std[i]])
        lin_obj = np.concatenate((lin_term[i], [-kappa]))
        obj += (1 / num_samples) * lin_obj @ sub_z_obj

    opt_model.setObjective(expr=obj, sense=GRB.MINIMIZE)

    ## add mu variable
    # opt_model._sub_z_mu = MVar.fromlist(sub_k.values())
    # opt_model._mu_coeff = lin_term

    return opt_model


def warm_start_from_candidate(
    # domain: Domain,
    # model: tuple[np.ndarray, float, float],
    # data: tuple[np.ndarray, np.ndarray],
    # kappa: float,
    # model_core: gp.Model,
    candidates: np.ndarray,
    domain: Domain,
    opt_model: gp.Model,
):
    """

    ."""
    gbm_model_dict = opt_model._gbm_models
    opt_model.NumStart = candidates.shape[0]
    opt_model.update()

    test_arr = []

    for start in range(opt_model.NumStart):
        opt_model.params.StartNumber = start
        opt_model.update()
        x = candidates[start]

        # set continuous variables
        for idx, var_name in enumerate(domain.inputs.get_keys()):
            opt_model.getVarByName(var_name).Start = x[idx]
        # TODO: set cat variables

        # set leaf variables
        for i, (gbm_name, gbm_model) in enumerate(gbm_model_dict.items()):
            gbm_model: GbmModel
            act_leaves_x = gbm_model.get_active_leaves(x)
            z_arr = []
            for key, z in opt_model._z_l.items():
                z_gbm_name, z_tree_idx, z_leaf_encoding = key
                if z_gbm_name != gbm_name:
                    continue

                z.Start = 1 if act_leaves_x[z_tree_idx] == z_leaf_encoding else 0

                # opt_model.addConstr(z == (1 if act_leaves_x[z_tree_idx] == z_leaf_encoding else 0))
                z_arr.append(1 if act_leaves_x[z_tree_idx] == z_leaf_encoding else 0)

            test_arr.append(z_arr)


def build_opt_model_from_gp(
    domain: Domain,
    gbm_model: GbmModel,
    tree_gp: LeafGP | LeafMOGP,
    kappa: float,
    model_core: Optional[gp.Model] = None,
):
    """Build an optimization model for the acquisition function"""
    # build opt_model core

    # check if there's already a model core with extra constraints
    if model_core is None:
        opt_model = get_opt_core(domain)
    else:
        # copy model core in case there are constr given already
        opt_model = get_opt_core_copy(model_core)

    # build tree model
    gbm_model_dict = {"1st_obj": gbm_model}
    cat_idx = get_cat_idx_from_domain(domain)
    add_gbm_to_opt_model(cat_idx, gbm_model_dict, opt_model)

    if isinstance(tree_gp, LeafMOGP):
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
            torch.Size(k_diag.shape), [torch.zeros((k_diag.shape[0], 1))]
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

    train_x: torch.Tensor
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

    lin_term = kernel_var * (inv_ks @ y_vals)

    lin_obj = np.concatenate((lin_term, [-kappa]))

    opt_model.setMObjective(
        None, lin_obj, 0, xc=opt_model._sub_z_obj, sense=GRB.MINIMIZE
    )

    ## add mu variable
    opt_model._sub_z_mu = MVar(sub_k.values())
    opt_model._mu_coeff = lin_term

    return opt_model
