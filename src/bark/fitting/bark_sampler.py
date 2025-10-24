from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from bofire.data_models.domain.api import Domain
from jaxtyping import Float

from bark import forest, types
from bark.fitting.noise_proposals import get_noise_scale_proposal
from bark.fitting.quick_inverse import LowRankInverter, mll
from bark.fitting.tree_proposals import get_forest_proposal
from bark.types import BARKModel
from bofire_mixed.domain import get_feature_bounds, get_feature_types_array


def run_bark_sampler(
    bark_model: BARKModel, data: types.DataT, domain: Domain, params: types.BARKConfig
) -> BARKModel:
    """Generate samples from the BARK posterior"""

    # unpack the model
    train_x, train_y = data

    # unpack the domain
    bounds = [
        get_feature_bounds(feat, encoding="bitmask") for feat in domain.inputs.get()
    ]

    bounds = np.array(bounds)

    feat_type = get_feature_types_array(domain)

    samples = _run_bark_sampler_multichain(
        bark_model, train_x, train_y, bounds, feat_type, params
    )

    return samples


def _run_bark_sampler_multichain(
    bark_model: BARKModel,
    train_x: Float[jax.Array, "N D"],
    train_y: Float[jax.Array, "N 1"],
    bounds: Float[jax.Array, "N 2"],
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
) -> BARKModel:
    num_chains = params.num_chains
    num_samples = params.num_samples

    assert bark_model.trees.threshold.shape[0] == num_chains
    # unstack the BARKModel
    # https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    leaves, treedef = jax.tree.flatten(bark_model)
    # initial_bark_models = [
    #     treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)
    # ]

    warmup_steps = params.warmup_steps
    steps_per_sample = params.steps_per_sample

    samples = []

    for chain_idx in range(num_chains):
        # initial values of K_inv and K_logdet
        K_XX = forest.forest_gram_matrix(
            train_x,
            train_x,
            bark_model.trees.feature_idx,
            bark_model.trees.threshold,
            feat_types,
        )
        K_XX_s = K_XX + (1e-6 + bark_model.noise) * np.eye(K_XX.shape[0])
        # In tests, using Cholesky solves does not seem to improve the speed of the
        # solver.
        cur_K_inv = np.linalg.inv(K_XX_s)
        _, cur_K_logdet = np.linalg.slogdet(K_XX_s)
        cur_mll = mll(cur_K_inv, cur_K_logdet, train_y)

        low_rank_inverter = LowRankInverter(
            K_inv=cur_K_inv,
            K_logdet=cur_K_logdet,
            mll=cur_mll,
            U=jnp.zeros((K_XX.shape[0], 1)),
            subtract=False,
            y=train_y,
        )

        for itr in range(warmup_steps + num_samples * steps_per_sample):
            (bark_model, low_rank_inverter) = _step_bark_sampler(
                bark_model,
                train_x,
                train_y,
                bounds,
                feat_types,
                params,
                low_rank_inverter,
            )
            step_itr = itr - warmup_steps
            if step_itr > 0 and step_itr % steps_per_sample == steps_per_sample - 1:
                samples.append(bark_model)

    bark_models_stacked = jax.tree.map(lambda *v: jnp.stack(v, axis=0), *samples)
    return bark_models_stacked


def _step_bark_sampler(
    bark_model: BARKModel,
    train_x: Float[jax.Array, "N D"],
    train_y: Float[jax.Array, "N 1"],
    bounds: Float[jax.Array, "N 2"],
    feat_types: types.FeatTypesT,
    params: types.BARKConfig,
    key: jax.Array,
    # low_rank_inverter: LowRankInverter,
) -> tuple[BARKModel, LowRankInverter]:
    m = bark_model.num_trees

    key, noise_key = jax.random.split(key)
    key, *proposal_key = jax.random.split(key, num=3)
    tree_key = jax.random.split(key, num=m)

    new_trees, tree_log_q_prior_ratio = get_forest_proposal(
        bark_model.trees, bounds, feat_types, params, tree_key
    )

    # invsqrtm = jnp.sqrt(1 / m)

    # cur_leaf_vectors = invsqrtm * forest.get_leaf_vectors(
    #     train_x,
    #     bark_model.trees.feature_idx[..., tree_idx, :],
    #     bark_model.trees.threshold[..., tree_idx, :],
    #     feat_types,
    # )
    # new_leaf_vectors = invsqrtm * forest.get_leaf_vectors(
    #     train_x,
    #     new_bark_model.forest.feature_idx[..., tree_idx, :],
    #     new_bark_model.forest.threshold[..., tree_idx, :],
    #     feat_types,
    # )

    # # compute the rank-one update for the inverse
    # lr_update: LowRankInverter = low_rank_inverter.set_low_rank_update_matrix(
    #     U=cur_leaf_vectors, subtract=True
    # ).low_rank_update()

    # lr_update: LowRankInverter = lr_update.set_low_rank_update_matrix(
    #     U=new_leaf_vectors, subtract=False
    # ).low_rank_update()

    # log_ll = lr_update.mll - low_rank_inverter.mll
    log_alpha = tree_log_q_prior_ratio + log_ll
    if np.log(np.random.uniform()) <= min(log_alpha, 0):
        # accept - set the new mll and K_inv values
        low_rank_inverter = lr_update
        bark_model = new_bark_model

    new_bark_model, log_q_prior = get_noise_scale_proposal(noise, params)
    K_XX = forest.forest_gram_matrix(
        train_x,
        train_x,
        bark_model.trees.feature_idx,
        bark_model.trees.threshold,
        feat_types,
    )
    K_XX_s = K_XX + (1e-6 + new_bark_model.noise) * np.eye(K_XX.shape[0])
    new_K_inv = jnp.linalg.inv(K_XX_s)
    _, new_K_logdet = jnp.linalg.slogdet(K_XX_s)

    new_mll = mll(new_K_inv, new_K_logdet, train_y)
    log_ll = new_mll - low_rank_inverter.mll
    log_alpha = log_q_prior + log_ll

    if np.log(np.random.uniform()) <= min(log_alpha, 0):
        # accept - set the new mll and K_inv values
        low_rank_inverter = replace(
            low_rank_inverter,
            K_inv=new_K_inv,
            K_logdet=new_K_logdet,
            mll=new_mll,
        )
        bark_model = new_bark_model

    return bark_model, low_rank_inverter
