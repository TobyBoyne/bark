import gurobipy as gp
import jax
import jax.numpy as jnp
import numpy as np
from gurobipy import GRB

from bark import forest, types
from bark.enums import FeatureTypeEnum
from bark.types.optimizer import GurobiOptimizerModel

from .mip_model import TreesMIPModel
from .opt_core import add_trees_to_opt_model, get_opt_core_copy


def build_opt_model_from_forest(
    bark_model: types.BARKModel,
    data: types.Data,
    kappa: float,
    model_core: GurobiOptimizerModel,
):
    opt_model = get_opt_core_copy(model_core)
    train_X, train_Y = data.train_X, data.train_Y
    train_Y = (train_Y - train_Y.mean()) / train_Y.std()

    # unpack samples
    bark_model = bark_model.get_flattened_samples()

    num_samples = bark_model.batch_shape[0]
    num_data = train_X.shape[0]

    # build tree model
    gbm_model_dict: dict[str, TreesMIPModel] = {}
    for sample_idx in range(num_samples):
        trees = jax.tree_util.tree_map(lambda t: t[sample_idx], bark_model.trees)
        gbm_model_dict[f"tree_sample_{sample_idx}"] = TreesMIPModel(
            trees, data.feat_types
        )

    cat_idx = {i for i, f in enumerate(data.feat_types) if f == FeatureTypeEnum.Cat}
    add_trees_to_opt_model(cat_idx, gbm_model_dict, opt_model)
    K_XX = jax.vmap(forest.forest_gram_matrix_no_null, in_axes=(None, 0, None))(
        train_X, bark_model.trees, data.feat_types
    )

    K_XX_s = K_XX + (1e-6 + bark_model.noise_var[:, None, None]) * np.eye(num_data)
    # cholesky decomposition doesn't support batching
    K_inv = np.asarray(jnp.linalg.inv(K_XX_s))

    sub_k = opt_model.addMVar(
        shape=(num_samples, num_data), lb=0, ub=1, name="sub_k", vtype="C"
    )
    for i, (gbm_name, gbm_model) in enumerate(gbm_model_dict.items()):
        # create active leaf variables
        act_leave_vars = gbm_model.get_active_leaf_vars(train_X, opt_model, gbm_name)
        opt_model.addConstrs(
            (sub_k[i, idx] == act_leave_vars[idx] for idx in range(num_data)),
            name=f"sub_k_constr_{gbm_name}",
        )

    ## add quadratic constraints
    # \sigma <= K_xx - K_xX @ K_XX^-1 @ X_xX^T

    opt_model._std = opt_model.addMVar(
        shape=(num_samples,), lb=0.0, ub=GRB.INFINITY, vtype="C"
    )

    # pre- and post-multiply by scale
    quadr_term = -K_inv
    const_term = jnp.ones_like(bark_model.noise_var)
    zeros = np.zeros((train_X.shape[0], 1))

    for i in range(num_samples):
        quadr_constr = np.block([[quadr_term[i], zeros], [zeros.T, -1.0]])
        sub_k_sample_std = gp.concatenate(
            (sub_k[i, :], opt_model._std[i][None]),  # type: ignore
            axis=0,
        )
        opt_model.addMQConstr(  # type: ignore
            quadr_constr,
            None,
            sense=">",
            rhs=-const_term[i],
            xQ_L=sub_k_sample_std,
            xQ_R=sub_k_sample_std,
        )

    ## add linear objective
    lin_term = K_inv @ train_Y[None, :, :]
    lin_term = lin_term.squeeze(-1)

    sub_z_obj = gp.concatenate((sub_k, opt_model._std[:, None]), axis=1)  # type: ignore
    lin_obj = np.concatenate((lin_term, np.full((num_samples, 1), -kappa)), axis=1)
    # compute row-wise dot product
    obj = (lin_obj * sub_z_obj).sum() / num_samples

    opt_model.setObjective(expr=obj, sense=GRB.MINIMIZE)

    ## add mu variable
    opt_model._sub_z_mu = sub_k
    opt_model._mu_coeff = lin_term

    return opt_model
