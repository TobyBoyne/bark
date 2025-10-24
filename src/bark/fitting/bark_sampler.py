import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from bark import types
from bark.fitting.marginal_log_likelihood import mll_bark_model
from bark.fitting.noise_proposals import get_noise_proposal_softplus
from bark.fitting.tree_proposals import get_forest_proposal
from bark.types import BARKModel


def run_bark_sampler(
    bark_model: BARKModel, data: types.Data, params: types.BARKConfig, seed: int
) -> BARKModel:
    """Generate samples from the BARK posterior"""

    num_samples = params.num_samples
    warmup_steps = params.warmup_steps
    steps_per_sample = params.steps_per_sample
    num_steps_total = warmup_steps + steps_per_sample * num_samples

    cur_mll = mll_bark_model(bark_model, data)
    # TODO: check that these keys are different across parallel chains
    keys = jax.random.split(jax.random.key(seed), num=num_steps_total)

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
    start_steps = jnp.arange(
        warmup_steps, num_steps_total - warmup_steps, step=steps_per_sample
    )
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

    def tree_propose_loop(
        i: int, val: tuple[BARKModel, Float[Array, ""]]
    ) -> tuple[BARKModel, Float[Array, ""]]:
        model, cur_mll = val
        selected_tree_mask = jnp.arange(m) == i
        selected_new_tree = jax.tree_util.tree_map(
            lambda t, nt: jnp.where(selected_tree_mask, nt, t), model.trees, new_trees
        )
        new_model = BARKModel(trees=selected_new_tree, noise=model.noise)

        new_mll = mll_bark_model(new_model, data)

        log_q_prior = tree_log_q_prior_ratio[i]
        log_ll = new_mll - cur_mll
        log_alpha = jnp.clip(log_q_prior + log_ll, min=0.0)

        accept = jnp.log(jax.random.uniform(proposal_key[1])) <= log_alpha
        # TODO: there has to be a better one to select one of the two models?
        # want to write `model = new_model if accept else model`
        model = BARKModel(
            trees=types.Tree(
                feature_idx=jnp.where(
                    accept, new_model.trees.feature_idx, model.trees.feature_idx
                ),
                threshold=jnp.where(
                    accept, new_model.trees.threshold, model.trees.threshold
                ),
            ),
            noise=jnp.where(accept, new_model.noise, model.noise),
        )
        cur_mll = jnp.where(accept, new_mll, cur_mll)
        return (model, cur_mll)

    bark_model, cur_mll = jax.lax.fori_loop(
        0, m, tree_propose_loop, (bark_model, cur_mll)
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

    if jnp.log(jax.random.uniform(proposal_key[1])) <= log_alpha:
        # accept - set the new mll and K_inv values
        cur_mll = new_mll
        bark_model = new_bark_model

    return bark_model, cur_mll
