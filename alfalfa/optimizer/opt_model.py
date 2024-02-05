import numpy as np

from ..tree_kernels import AlfalfaGP
from ..utils.space import Space
from .gbm_model import GbmModel
from .optimizer_utils import (add_gbm_to_opt_model, get_opt_core)


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
    gbm_model_dict = {"1st_obj": gbm_model}
    add_gbm_to_opt_model(space, gbm_model_dict, opt_model, z_as_bin=True)

    # get tree_gp hyperparameters
    kernel_var = tree_gp.covar_module.outputscale.detach().numpy()
    noise_var = tree_gp.likelihood.noise.detach().numpy()

    # get tree_gp matrices
    (train_x,) = tree_gp.train_inputs
    Kmm = tree_gp.covar_module(train_x).numpy()
    k_diag = np.diagonal(Kmm)
    s_diag = tree_gp.likelihood._shaped_noise_covar(k_diag.shape).numpy()
    ks = Kmm + s_diag

    # invert gram matrix
    from scipy.linalg import cho_factor, cho_solve

    id_ks = np.eye(ks.shape[0])
    inv_ks = cho_solve(cho_factor(ks, lower=True), id_ks)

    # add tree_gp logic to opt_model
    from gurobipy import GRB, MVar

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
    opt_model._var = opt_model.addVar(lb=0, ub=GRB.INFINITY, name="var", vtype="C")

    opt_model._sub_k_var = MVar(
        [sub_k[id] for id in range(len(sub_k))] + [opt_model._var]
    )

    quadr_term = -(kernel_var**2) * inv_ks
    const_term = kernel_var + noise_var

    quadr_constr = np.zeros((quadr_term.shape[0] + 1, quadr_term.shape[1] + 1))
    quadr_constr[:-1, :-1] = quadr_term
    quadr_constr[-1, -1] = -1.0

    opt_model.addMQConstr(
        quadr_constr,
        None,
        sense=">",
        rhs=-const_term,
        xQ_L=opt_model._sub_k_var,
        xQ_R=opt_model._sub_k_var,
    )

    ## add linear objective
    opt_model._sub_z_obj = MVar(
        [sub_k[idx] for idx in range(len(sub_k))] + [opt_model._var]
    )

    y_vals = tree_gp.train_targets.numpy().tolist()
    lin_term = kernel_var * np.matmul(inv_ks, np.asarray(y_vals))

    lin_obj = np.zeros(len(lin_term) + 1)
    lin_obj[:-1] = lin_term
    lin_obj[-1] = -kappa

    opt_model.setMObjective(
        None, lin_obj, 0, xc=opt_model._sub_z_obj, sense=GRB.MINIMIZE
    )

    ## add mu variable
    opt_model._sub_z_mu = MVar([sub_k[idx] for idx in range(len(sub_k))])
    opt_model._mu_coeff = lin_term

    return opt_model