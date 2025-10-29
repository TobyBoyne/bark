from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bark import types
from bark.fitting.marginal_log_likelihood import (
    get_cached_grams,
    mll_bark_model,
    mll_bark_model_cached_gram,
)
from bark.fitting.noise_proposals import get_noise_proposal_softplus
from bark.fitting.tree_proposals import get_forest_proposal
from bark.types import BARKModel


def run_bark_sampler(
    bark_model: BARKModel, data: types.Data, params: types.BARKConfig, key: jax.Array
) -> BARKModel:
    """Generate samples from the BARK posterior"""

    if params.num_chains == 1:
        return _run_bark_sampler(bark_model, data, params, key)

    if not bark_model.batch_shape:
        # bark model is a single sample, must create copies
        bark_model = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, reps=(params.num_chains, *[1 for _ in x.shape])),
            bark_model,
        )
    elif bark_model.batch_shape[-1] != params.num_chains:
        raise ValueError(
            "The trailing batch dimension must be equal to the number of parallel "
            f"chains (expected {params.num_chains}, got {bark_model.batch_shape[-1]})"
        )

    keys = jax.random.split(key, params.num_chains)
    return jax.vmap(_run_bark_sampler, in_axes=(0, None, None, 0))(
        bark_model,
        data,
        params,
        keys,
    )


@partial(jax.jit, static_argnames=("params"))
def _run_bark_sampler(
    bark_model: BARKModel, data: types.Data, params: types.BARKConfig, key: jax.Array
) -> BARKModel:
    num_samples = params.num_samples
    warmup_steps = params.warmup_steps
    steps_per_sample = params.steps_per_sample
    num_steps_total = warmup_steps + steps_per_sample * num_samples

    cur_mll = mll_bark_model(bark_model, data)
    keys = jax.random.split(key, num=num_steps_total)

    def step_bark(
        i: int, val: tuple[BARKModel, Float[Array, ""]]
    ) -> tuple[BARKModel, Float[Array, ""]]:
        bark_model, _cur_mll = val
        return _step_bark_sampler(bark_model, data, params, keys[i])

    def sample_bark(
        carry: tuple[BARKModel, Float[Array, ""]], x: Int[Array, ""]
    ) -> tuple[tuple[BARKModel, Float[Array, ""]], BARKModel]:
        carry = jax.lax.fori_loop(
            lower=x, upper=x + steps_per_sample, body_fun=step_bark, init_val=carry
        )
        bark_model = carry[0]
        return carry, bark_model

    # generate warmup samples
    carry = jax.lax.fori_loop(
        lower=0, upper=warmup_steps, body_fun=step_bark, init_val=(bark_model, cur_mll)
    )

    # generate samples
    start_steps = jnp.arange(warmup_steps, num_steps_total, step=steps_per_sample)
    _carry, samples = jax.lax.scan(sample_bark, carry, start_steps)

    return samples


def _step_bark_sampler(
    bark_model: BARKModel,
    data: types.Data,
    params: types.BARKConfig,
    key: jax.Array,
    # low_rank_inverter: LowRankInverter,
) -> tuple[BARKModel, Float[Array, ""]]:
    m = bark_model.num_trees
    # TODO: pass cur_mll from previous iteration
    cur_mll = mll_bark_model(bark_model, data)
    key, noise_key = jax.random.split(key)
    key, *proposal_key = jax.random.split(key, num=3)
    tree_key = jax.random.split(key, num=m)

    new_trees, tree_log_q_prior_ratio = get_forest_proposal(
        bark_model.trees, data.bounds, data.feat_types, params, tree_key
    )
    G_XX_init, G_XX_delta = get_cached_grams(bark_model.trees, new_trees, data)

    def tree_propose_loop(
        i: int, val: tuple[BARKModel, Float[Array, ""], Float[Array, "N N"]]
    ) -> tuple[BARKModel, Float[Array, ""], Float[Array, "N N"]]:
        model, cur_mll, G_XX = val
        new_mll = mll_bark_model_cached_gram(bark_model, G_XX, G_XX_delta[..., i], data)

        log_q_prior = tree_log_q_prior_ratio[i]
        log_ll = new_mll - cur_mll
        log_alpha = jnp.clip(log_q_prior + log_ll, min=0.0)

        accept = jnp.log(jax.random.uniform(proposal_key[1])) <= log_alpha
        model = model.update_trees(new_trees, accept)
        cur_mll = jnp.where(accept, new_mll, cur_mll)
        G_XX = jnp.where(accept, G_XX + G_XX_delta[..., i], G_XX)
        return (model, cur_mll, G_XX)

    bark_model, cur_mll, _ = jax.lax.fori_loop(
        0, m, tree_propose_loop, (bark_model, cur_mll, G_XX_init)
    )

    new_noise, log_q_prior = get_noise_proposal_softplus(
        bark_model.noise, params, noise_key
    )
    new_bark_model = BARKModel(
        trees=bark_model.trees,
        noise=new_noise,
    )
    new_mll = mll_bark_model(bark_model, data)

    log_ll = new_mll - cur_mll
    log_alpha = jnp.clip(log_q_prior + log_ll, min=0.0)

    accept = jnp.log(jax.random.uniform(proposal_key[1])) <= log_alpha
    bark_model = bark_model.update_noise(new_bark_model.noise, accept)
    cur_mll = jnp.where(accept, new_mll, cur_mll)

    return bark_model, cur_mll  # pyright: ignore
