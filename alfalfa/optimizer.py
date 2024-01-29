import numpy as np
import torch
from .leaf_gp.optimizer_utils import \
    get_opt_core, add_gbm_to_opt_model, get_opt_core_copy, label_leaf_index
from .leaf_gp.space import Space
from .leaf_gp.gbm_model import GbmModel
from .tree_models.tree_kernels import AlfalfaGP

import gurobipy

def build_opt_model(space: Space, gbm_model: GbmModel, tree_gp: AlfalfaGP, kappa):
    # build opt_model core

    # # check if there's already a model core with extra constraints
    # if self.model_core is None:
    #     opt_model = get_opt_core(self.space)
    # else:
    #     # copy model core in case there are constr given already
    #     opt_model = get_opt_core_copy(self.model_core)
    opt_model = get_opt_core(space)


    # build tree model
    gbm_model_dict = {'1st_obj': gbm_model}
    add_gbm_to_opt_model(space,
                            gbm_model_dict,
                            opt_model,
                            z_as_bin=True)

    # get tree_gp hyperparameters
    kernel_var = tree_gp.covar_module.outputscale.detach().numpy()
    noise_var = tree_gp.likelihood.noise.detach().numpy()

    # get tree_gp matrices
    train_x, = tree_gp.train_inputs
    Kmm = tree_gp.covar_module(train_x).numpy()
    k_diag = np.diagonal(Kmm)
    s_diag = tree_gp.likelihood._shaped_noise_covar(k_diag.shape).numpy()
    ks = Kmm + s_diag

    # invert gram matrix
    from scipy.linalg import cho_solve, cho_factor
    id_ks = np.eye(ks.shape[0])
    inv_ks = cho_solve(
        cho_factor(ks, lower=True), id_ks)

    # add tree_gp logic to opt_model
    from gurobipy import GRB, MVar

    act_leave_vars = gbm_model.get_active_leaf_vars(
        train_x.numpy(),
        opt_model,
        '1st_obj')

    sub_k = opt_model.addVars(
        range(len(act_leave_vars)),
        lb=0,
        ub=1,
        name="sub_k",
        vtype='C')

    opt_model.addConstrs(
        (sub_k[idx] == act_leave_vars[idx]
            for idx in range(len(act_leave_vars))),
        name='sub_k_constr')

    ## add quadratic constraints
    opt_model._var = opt_model.addVar(
        lb=0,
        ub=GRB.INFINITY,
        name="var",
        vtype='C')

    opt_model._sub_k_var = MVar(
        [sub_k[id]
            for id in range(len(sub_k))] + [opt_model._var])

    quadr_term = - (kernel_var ** 2) * inv_ks
    const_term = kernel_var + noise_var

    quadr_constr = np.zeros(
        (quadr_term.shape[0] + 1,
        quadr_term.shape[1] + 1))
    quadr_constr[:-1, :-1] = quadr_term
    quadr_constr[-1, -1] = -1.0

    opt_model.addMQConstr(
        quadr_constr, None,
        sense='>',
        rhs=-const_term,
        xQ_L=opt_model._sub_k_var,
        xQ_R=opt_model._sub_k_var)

    ## add linear objective
    opt_model._sub_z_obj = MVar(
        [sub_k[idx] for idx in range(len(sub_k))] + [opt_model._var])

    y_vals = tree_gp.train_targets.numpy().tolist()
    lin_term = kernel_var * np.matmul(
        inv_ks, np.asarray(y_vals))

    lin_obj = np.zeros(len(lin_term) + 1)
    lin_obj[:-1] = lin_term
    lin_obj[-1] = - kappa

    opt_model.setMObjective(
        None, lin_obj, 0,
        xc=opt_model._sub_z_obj,
        sense=GRB.MINIMIZE)

    ## add mu variable
    opt_model._sub_z_mu = MVar(
        [sub_k[idx] for idx in range(len(sub_k))])
    opt_model._mu_coeff = lin_term

    return opt_model

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
    from .leaf_gp.optimizer_utils import get_opt_sol
    next_x = get_opt_sol(space, opt_model)

    # extract variance and mean
    curr_var = opt_model._var.x
    curr_mean = sum([opt_model._mu_coeff[idx]*opt_model._sub_z_mu[idx].x
                        for idx in range(len(opt_model._mu_coeff))])

    return var_bnds, next_x, curr_mean, curr_var