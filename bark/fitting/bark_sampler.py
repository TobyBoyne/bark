from dataclasses import dataclass

import numpy as np
import tqdm
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, DiscreteInput
from numba import njit, prange

from bark.fitting.noise_scale_proposals import get_noise_scale_proposal
from bark.fitting.quick_inverse import low_rank_det_update, low_rank_inv_update, mll
from bark.fitting.tree_proposals import FeatureTypeEnum, get_tree_proposal
from bark.forest import NODE_RECORD_DTYPE, forest_gram_matrix, get_leaf_vectors
from bark.utils.domain import get_feature_bounds

ModelT = tuple[np.ndarray, float, float]
DataT = tuple[np.ndarray, np.ndarray]


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

    num_chains: int = 1
    verbose: bool = False

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
        ("num_chains", np.uint32),
        ("alpha", np.float32),
        ("beta", np.float32),
        ("proposal_weights", np.float32, (3,)),
        ("verbose", np.bool_),
    ]
)


def _bark_params_to_struct(params: BARKTrainParams):
    return np.record(
        (
            params.warmup_steps,
            params.n_steps,
            params.thinning,
            params.num_chains,
            params.alpha,
            params.beta,
            params.proposal_weights,
            params.verbose,
        ),
        dtype=BARKTRAINPARAMS_DTYPE,
    )


def run_bark_sampler(
    model: ModelT, data: DataT, domain: Domain, params: BARKTrainParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate samples from the BARK posterior"""

    # unpack the model
    train_x, train_y = data
    forest, noise, scale = model

    # unpack the domain
    bounds = [
        get_feature_bounds(feat, ordinal_encoding=True) for feat in domain.inputs.get()
    ]

    feat_type = np.array(
        [
            FeatureTypeEnum.Cat.value
            if isinstance(feat, CategoricalInput)
            else FeatureTypeEnum.Int.value
            if isinstance(feat, DiscreteInput)
            else FeatureTypeEnum.Cont.value
            for feat in domain.inputs.get()
        ]
    )

    params_struct = _bark_params_to_struct(params)

    samples = _run_bark_sampler_multichain(
        forest, noise, scale, train_x, train_y, bounds, feat_type, params_struct
    )

    return samples


# @njit
def _run_bark_sampler_multichain(
    forest: np.ndarray,
    noise: np.ndarray,
    scale: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: list[list[float]],
    feat_types: np.ndarray,
    params: np.record,
):
    # this function can't be jitted nor parallelized - possibly due to rng
    np.random.seed(42)
    num_chains = params["num_chains"]
    num_samples = int(np.ceil(params["n_steps"] / params["thinning"]))

    node_samples = np.zeros(
        (num_chains, num_samples, *forest.shape[-2:]), dtype=NODE_RECORD_DTYPE
    )
    noise_samples = np.zeros((num_chains, num_samples), dtype=np.float32)
    scale_samples = np.zeros((num_chains, num_samples), dtype=np.float32)

    for chain_idx in prange(num_chains):
        node_s, noise_s, scale_s = _run_bark_sampler(
            forest[chain_idx],
            noise[chain_idx],
            scale[chain_idx],
            train_x,
            train_y,
            bounds,
            feat_types,
            params,
        )

        node_samples[chain_idx] = node_s
        noise_samples[chain_idx] = noise_s
        scale_samples[chain_idx] = scale_s

    return (node_samples, noise_samples, scale_samples)


# @njit
def _run_bark_sampler(
    forest: np.ndarray,
    noise: float,
    scale: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: list[list[float]],
    feat_types: np.ndarray,
    params: np.record,
):
    # forest is (m x N) array of nodes
    num_samples = params["n_steps"] // params["thinning"]
    node_samples = np.zeros((num_samples, *forest.shape), dtype=NODE_RECORD_DTYPE)
    noise_samples = np.zeros(num_samples, dtype=np.float32)
    scale_samples = np.zeros(num_samples, dtype=np.float32)

    # initial values of K_inv and K_logdet
    K_XX = scale * forest_gram_matrix(forest, train_x, train_x, feat_types)
    K_XX_s = K_XX + noise * np.eye(K_XX.shape[0])
    # TODO: should use cholesky
    # https://numba.discourse.group/t/how-can-i-improve-the-runtime-of-this-linear-system-solve/2406
    # https://github.com/numba/numba-scipy/issues/91
    cur_K_inv = np.linalg.inv(K_XX_s)
    _, cur_K_logdet = np.linalg.slogdet(K_XX_s)
    cur_mll = mll(cur_K_inv, cur_K_logdet, train_y)

    for itr in tqdm.tqdm(
        range(params["warmup_steps"]), desc="Warmup", disable=not params["verbose"]
    ):
        forest, noise, scale, cur_K_inv, cur_K_logdet, cur_mll = _step_bark_sampler(
            forest,
            noise,
            scale,
            train_x,
            train_y,
            bounds,
            feat_types,
            params,
            cur_K_inv,
            cur_K_logdet,
            cur_mll,
        )

    for itr in tqdm.tqdm(
        range(params["n_steps"]), desc="Sampling", disable=not params["verbose"]
    ):
        forest, noise, scale, cur_K_inv, cur_K_logdet, cur_mll = _step_bark_sampler(
            forest,
            noise,
            scale,
            train_x,
            train_y,
            bounds,
            feat_types,
            params,
            cur_K_inv,
            cur_K_logdet,
            cur_mll,
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
    noise: float,
    scale: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: list[list[float]],
    feat_types: np.ndarray,
    params: np.record,
    cur_K_inv: np.ndarray,
    cur_K_logdet: float,
    cur_mll: float,
):
    m = forest.shape[0]
    s_sqrtm = np.sqrt(scale / m)

    for tree_idx in range(m):
        new_nodes, log_q_prior = get_tree_proposal(
            forest[tree_idx], bounds, feat_types, params
        )

        cur_leaf_vectors = s_sqrtm * get_leaf_vectors(
            forest[tree_idx], train_x, feat_types
        )
        new_leaf_vectors = s_sqrtm * get_leaf_vectors(new_nodes, train_x, feat_types)

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
        if np.log(np.random.uniform()) <= min(log_alpha, 0):
            # accept - set the new mll and K_inv values
            cur_K_inv = new_K_inv
            cur_K_logdet = new_K_logdet
            cur_mll = new_mll
            forest[tree_idx] = new_nodes

    (new_noise, new_scale), log_q_prior = get_noise_scale_proposal(noise, scale)
    K_XX = new_scale * forest_gram_matrix(forest, train_x, train_x, feat_types)
    K_XX_s = K_XX + new_noise * np.eye(K_XX.shape[0])
    new_K_inv = np.linalg.inv(K_XX_s)
    _, new_K_logdet = np.linalg.slogdet(K_XX_s)

    new_mll = mll(new_K_inv, new_K_logdet, train_y)
    log_ll = new_mll - cur_mll
    log_alpha = log_q_prior + log_ll
    # print(">>")
    # print(noise, scale)
    # print(new_noise, new_scale)
    # print(log_q_prior, log_ll)
    if np.log(np.random.uniform()) <= min(log_alpha, 0):
        noise = new_noise
        cur_K_inv = new_K_inv
        cur_K_logdet = new_K_logdet
        cur_mll = new_mll
        scale = new_scale
        noise = new_noise

    return forest, noise, scale, cur_K_inv, cur_K_logdet, cur_mll
