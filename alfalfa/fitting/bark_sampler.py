from dataclasses import dataclass

import numpy as np
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, DiscreteInput
from numba import njit

from alfalfa.fitting.noise_scale_proposals import get_noise_proposal
from alfalfa.fitting.quick_inverse import low_rank_det_update, low_rank_inv_update, mll
from alfalfa.fitting.tree_proposals import FeatureTypeEnum, get_tree_proposal
from alfalfa.forest_numba import NODE_RECORD_DTYPE, get_leaf_vectors, pass_through_tree
from alfalfa.tree_kernels.tree_gps import AlfalfaGP
from alfalfa.utils.domain import get_feature_bounds


@dataclass
class BARKTrainParams:
    # MCMC run parameters
    warmup_steps: int = 50
    n_steps: int = 50
    thinning: int = 5

    # node depth prior
    alpha: float = 0.95
    beta: float = 2.0

    # transition type probabilities
    grow_prune_weight: float = 0.5
    change_weight: float = 1.0

    verbose: bool = True

    @property
    def proposal_weights(self):
        p = np.array(
            [self.grow_prune_weight, self.grow_prune_weight, self.change_weight]
        )
        return p / np.sum(p)


BARKTRAINPARAMS_DTYPE = np.dtype(
    [
        ("warmup_steps", np.uint32),
        ("n_steps", np.uint32),
        ("thinning", np.uint32),
        ("alpha", np.float32),
        ("beta", np.float32),
        ("proposal_weights", np.float32, (3,)),
    ]
)


def _bark_params_to_struct(params: BARKTrainParams):
    return np.array(
        (
            params.warmup_steps,
            params.n_steps,
            params.thinning,
            params.alpha,
            params.beta,
            params.proposal_weights,
        ),
        dtype=BARKTRAINPARAMS_DTYPE,
    )


def run_bark_sampler(model: AlfalfaGP, domain: Domain, params: BARKTrainParams):
    """Generate samples from the BARK posterior"""

    # unpack the model
    (train_x,) = model.train_inputs
    train_y = model.train_targets
    forest = model.tree_model

    noise = model.likelihood.noise.item()
    scale = model.covar_module.outputscale.item()

    # unpack the domain
    bounds = [
        get_feature_bounds(feat, ordinal_encoding=True) for feat in domain.inputs.get()
    ]
    feat_type = np.array(
        [
            FeatureTypeEnum.Cat
            if isinstance(feat, CategoricalInput)
            else FeatureTypeEnum.Int
            if isinstance(feat, DiscreteInput)
            else FeatureTypeEnum.Cont
            for feat in domain.inputs.get()
        ]
    )

    params_struct = _bark_params_to_struct(params)

    samples = _run_bark_sampler(
        forest, train_x, train_y, bounds, feat_type, noise, scale, params_struct
    )

    return samples


@njit
def _run_bark_sampler(
    forest: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: list[list[float]],
    feat_types: np.ndarray,
    noise: float,
    scale: float,
    params: np.ndarray,
):
    # forest is (m x N) array of nodes
    rng = np.random.default_rng(seed=42)
    num_samples = params["n_steps"] // params["thinning"]
    node_samples = np.zeros((num_samples, *forest.shape), dtype=NODE_RECORD_DTYPE)
    noise_samples = np.zeros(num_samples, dtype=np.float32)
    scale_samples = np.zeros(num_samples, dtype=np.float32)
    for itr in range(params["warmup_steps"]):
        forest, noise, scale = _step_bark_sampler(
            forest, train_x, train_y, bounds, feat_types, noise, scale, rng, params
        )

    for itr in range(params["n_steps"]):
        forest, noise, scale = _step_bark_sampler(
            forest, train_x, train_y, bounds, feat_types, noise, scale, rng, params
        )
        if itr % params["thinning"] == 0:
            slice_idx = itr // params["thinning"]
            node_samples[slice_idx, :, :] = forest
            noise_samples[slice_idx] = noise
            scale_samples[slice_idx] = scale

    return node_samples, noise_samples, scale_samples


@njit
def _step_bark_sampler(
    forest: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: list[list[float]],
    feat_types: np.ndarray,
    noise: float,
    scale: float,
    rng: np.random.Generator,
    params: np.ndarray,
):
    # TODO: pass in previous K_inv and K_logdet
    m = forest.shape[0]
    K_XX = (scale / m) * pass_through_tree(forest, train_x, feat_types)
    K_XX_s = K_XX + noise * np.eye(K_XX.shape[0])
    # TODO: should use cholesky
    # https://numba.discourse.group/t/how-can-i-improve-the-runtime-of-this-linear-system-solve/2406
    # https://github.com/numba/numba-scipy/issues/91
    cur_K_inv = np.linalg.inv(K_XX_s)
    _, cur_K_logdet = np.linalg.slogdet(K_XX_s)
    cur_mll = mll(cur_K_inv, cur_K_logdet, train_y)
    for tree_idx in range(m):
        cur_nodes = forest[tree_idx].copy()
        new_nodes, log_q_prior = get_tree_proposal(
            forest[tree_idx], bounds, feat_types, rng, params
        )

        cur_leaf_vectors = get_leaf_vectors(forest, train_x, feat_types)
        forest[tree_idx] = new_nodes
        new_leaf_vectors = get_leaf_vectors(forest, train_x, feat_types)

        new_K_inv, new_K_logdet = (
            low_rank_inv_update(cur_K_inv, cur_leaf_vectors, subtract=True),
            low_rank_det_update(
                cur_K_inv, cur_leaf_vectors, cur_K_logdet, subtract=True
            ),
        )

        # K = torch.linalg.inv(K_inv)
        # actual = torch.linalg.inv(K + cur_leaf_vectors)

        new_K_inv, new_K_logdet = (
            low_rank_inv_update(new_K_inv, new_leaf_vectors),
            low_rank_det_update(new_K_inv, new_leaf_vectors, new_K_logdet),
        )
        new_mll = mll(new_K_inv, new_K_logdet, train_y)
        log_ll = new_mll - cur_mll
        log_alpha = log_q_prior + log_ll
        if np.log(rng.uniform()) <= min(log_alpha, 0):
            # accept - set the new mll and K_inv values
            cur_K_inv = new_K_inv
            cur_K_logdet = new_K_logdet
            cur_mll = new_mll
        else:
            # reject - reset the change
            forest[tree_idx] = cur_nodes

    (new_noise, new_scale), log_q_prior = get_noise_proposal(forest, noise, scale, rng)
    K_XX = (new_scale / m) * pass_through_tree(forest, train_x, feat_types)
    K_XX_s = K_XX + new_noise * np.eye(K_XX.shape[0])
    new_K_inv = np.linalg.inv(K_XX_s)
    _, new_K_logdet = np.linalg.slogdet(K_XX_s)

    new_mll = mll(cur_K_inv, cur_K_logdet, train_y)
    log_alpha = log_q_prior + log_ll

    if np.log(rng.uniform()) <= log_alpha:
        noise = new_noise
        cur_K_inv = new_K_inv
        cur_K_logdet = new_K_logdet
        cur_mll = new_mll
        scale = new_scale
        noise = new_noise

    return forest, noise, scale
