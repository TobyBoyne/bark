import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from bark import types
from bark.fitting.bark_prior_sampler import sample_forest
from bark.optimizer.build_opt_model import build_opt_model_from_forest
from bark.optimizer.opt_core import get_opt_core_from_bark_data
from bark.optimizer.proposals import propose
from bark.testing.data_test_cases import get_continuous_data_trid
from bark.tree_kernels.tree_gps import forest_predict

jax.config.update("jax_enable_x64", True)


def calculate_acqf(
    mu: Float[Array, "batch M"], var: Float[Array, "batch M"], kappa: float
) -> Float[Array, " M"]:
    std = jnp.sqrt(var)
    acqf = mu - kappa * std
    return acqf.mean(axis=-2)


def test_proposal_maximises_acqf():
    data = get_continuous_data_trid(N=20, dim=4)

    model_core = get_opt_core_from_bark_data(data)

    params = types.BARKConfig()
    keys = jax.random.split(jax.random.key(0), 10)
    trees = jax.vmap(sample_forest, in_axes=(None, None, None, None, 0))(
        50, data.bounds, data.feat_types, params, keys
    )
    bark_model = types.BARKModel(trees=trees, noise_var=jnp.full((10,), 0.1))

    opt_model = build_opt_model_from_forest(
        bark_model=bark_model,
        data=data,
        kappa=1.96,
        model_core=model_core,
    )

    features = types.Features(bounds=data.bounds, feat_types=data.feat_types)

    next_X = propose(features, opt_model, model_core)
    next_X_candidate = jnp.array([next_X])
    key_candidates = jax.random.key(92103)
    candidates = jax.random.uniform(key_candidates, (1000, data.train_X.shape[-1]))

    mu, var = forest_predict(bark_model, data, candidates, diag=True)
    acqf = calculate_acqf(mu, var, kappa=1.96)

    mux, varx = forest_predict(bark_model, data, next_X_candidate, diag=True)
    acqfx = calculate_acqf(mux, varx, kappa=1.96)

    assert acqfx.item() <= acqf.min()

    # test that the mu, var match between the model and the optimizer
    cand_mu, cand_var = forest_predict(bark_model, data, next_X_candidate, diag=True)

    cand_mu_opt = (
        (opt_model._mu_coeff * opt_model._sub_z_mu.X.reshape(opt_model._mu_coeff.shape))
        .sum(axis=-1)
        .mean()
    )
    cand_var_opt = jnp.array([s.X**2 for s in opt_model._std.values()])

    assert jnp.isclose(cand_mu.mean(), cand_mu_opt.mean(), rtol=0.1)
    assert jnp.isclose(cand_var.mean(), cand_var_opt.mean(), rtol=0.05)
