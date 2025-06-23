import numba as nb
import numpy as np
from bofire.data_models.domain.api import Domain
from numba import njit, prange
from numba.experimental import jitclass

from bark.fitting.noise_scale_proposals import get_noise_scale_proposal
from bark.fitting.quick_inverse import low_rank_det_update, low_rank_inv_update, mll
from bark.fitting.tree_proposals import get_tree_proposal
from bark.forest import NODE_RECORD_DTYPE, forest_gram_matrix, get_leaf_vectors
from bofire_mixed.domain import get_feature_bounds, get_feature_types_array

ModelT = tuple[np.ndarray, float, float]
DataT = tuple[np.ndarray, np.ndarray]


# @dataclass
# class BARKTrainParams:
#     # MCMC run parameters
#     warmup_steps: int = 50
#     num_samples: int = 5
#     steps_per_sample: int = 10

#     # noise and scale proposal parameters
#     use_softplus_transform: bool = True
#     sample_scale: bool = False
#     gamma_prior_shape: float = 2.5
#     gamma_prior_rate: float = 9.0

#     # node depth prior
#     alpha: float = 0.95
#     beta: float = 2.0

#     # transition type probabilities
#     grow_prune_weight: float = 0.5
#     change_weight: float = 1.0

#     num_chains: int = 1
#     verbose: bool = False

#     @property
#     def proposal_weights(self):
#         p = np.array(
#             [self.grow_prune_weight, self.grow_prune_weight, self.change_weight]
#         )
#         return p / np.sum(p)

BARK_JITCLASS_SPEC = [
    ("warmup_steps", nb.int64),
    ("num_samples", nb.int64),
    ("steps_per_sample", nb.int64),
    ("num_chains", nb.int64),
    ("alpha", nb.float64),
    ("beta", nb.float64),
    ("proposal_weights", nb.float64[:]),
    ("verbose", nb.bool_),
    ("use_softplus_transform", nb.bool_),
    ("sample_scale", nb.bool_),
    ("gamma_prior_shape", nb.float64),
    ("gamma_prior_rate", nb.float64),
]


@jitclass(BARK_JITCLASS_SPEC)
class BARKTrainParamsNumba:
    def __init__(
        self,
        warmup_steps,
        num_samples,
        steps_per_sample,
        num_chains,
        alpha,
        beta,
        proposal_weights,
        verbose,
        use_softplus_transform,
        sample_scale,
        gamma_prior_shape,
        gamma_prior_rate,
    ):
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.steps_per_sample = steps_per_sample
        self.num_chains = num_chains
        self.alpha = alpha
        self.beta = beta
        self.proposal_weights = proposal_weights
        self.verbose = verbose
        self.use_softplus_transform = use_softplus_transform
        self.sample_scale = sample_scale
        self.gamma_prior_shape = gamma_prior_shape
        self.gamma_prior_rate = gamma_prior_rate


def run_bark_sampler(
    model: ModelT, data: DataT, domain: Domain, params: BARKTrainParamsNumba
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate samples from the BARK posterior"""

    # unpack the model
    train_x, train_y = data
    forest, noise, scale = model

    # unpack the domain
    bounds = [
        get_feature_bounds(feat, encoding="bitmask") for feat in domain.inputs.get()
    ]

    bounds = np.array(bounds)

    feat_type = get_feature_types_array(domain)

    samples = _run_bark_sampler_multichain(
        forest, noise, scale, train_x, train_y, bounds, feat_type, params
    )

    return samples


@njit(parallel=False)
def _run_bark_sampler_multichain(
    forest: np.ndarray,
    noise: np.ndarray,
    scale: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: np.ndarray,
    feat_types: np.ndarray,
    params: BARKTrainParamsNumba,
):
    num_chains = params.num_chains
    num_samples = params.num_samples

    s = num_chains * num_samples * forest.shape[-2] * forest.shape[-1]
    node_samples_flat = np.empty((s), dtype=NODE_RECORD_DTYPE)

    node_samples = node_samples_flat.reshape(
        (num_chains, num_samples, *forest.shape[-2:])
    )

    noise_samples = np.empty((num_chains, num_samples), dtype=np.float64)
    scale_samples = np.empty((num_chains, num_samples), dtype=np.float64)

    warmup_steps = params.warmup_steps
    steps_per_sample = params.steps_per_sample

    for chain_idx in prange(num_chains):
        forest_chain = forest[chain_idx, :, :]
        noise_chain = noise[chain_idx]
        scale_chain = scale[chain_idx]

        # initial values of K_inv and K_logdet
        K_XX = scale_chain * forest_gram_matrix(
            forest_chain, train_x, train_x, feat_types
        )
        K_XX_s = K_XX + (1e-6 + noise_chain) * np.eye(K_XX.shape[0])
        # should use cholesky
        # https://numba.discourse.group/t/how-can-i-improve-the-runtime-of-this-linear-system-solve/2406
        # https://github.com/numba/numba-scipy/issues/91
        cur_K_inv = np.linalg.inv(K_XX_s)
        _, cur_K_logdet = np.linalg.slogdet(K_XX_s)
        cur_mll = mll(cur_K_inv, cur_K_logdet, train_y)

        for warmup_itr in range(warmup_steps):
            (
                forest_chain,
                noise_chain,
                scale_chain,
                cur_K_inv,
                cur_K_logdet,
                cur_mll,
            ) = _step_bark_sampler(
                forest_chain,
                noise_chain,
                scale_chain,
                train_x,
                train_y,
                bounds,
                feat_types,
                params,
                cur_K_inv,
                cur_K_logdet,
                cur_mll,
            )

        for itr in range(num_samples):
            for step in range(steps_per_sample):
                (
                    forest_chain,
                    noise_chain,
                    scale_chain,
                    cur_K_inv,
                    cur_K_logdet,
                    cur_mll,
                ) = _step_bark_sampler(
                    forest_chain,
                    noise_chain,
                    scale_chain,
                    train_x,
                    train_y,
                    bounds,
                    feat_types,
                    params,
                    cur_K_inv,
                    cur_K_logdet,
                    cur_mll,
                )

            node_samples[chain_idx, itr] = forest_chain[:, :]
            noise_samples[chain_idx, itr] = noise_chain
            scale_samples[chain_idx, itr] = scale_chain

    return (node_samples, noise_samples, scale_samples)


@njit
def _step_bark_sampler(
    forest: np.ndarray,
    noise: float,
    scale: float,
    train_x: np.ndarray,
    train_y: np.ndarray,
    bounds: np.ndarray,
    feat_types: np.ndarray,
    params: BARKTrainParamsNumba,
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

        # compute the rank-one update for the inverse
        new_K_inv, new_K_logdet = (
            low_rank_inv_update(cur_K_inv, cur_leaf_vectors, subtract=True),
            low_rank_det_update(
                cur_K_inv, cur_leaf_vectors, cur_K_logdet, subtract=True
            ),
        )

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

    (new_noise, new_scale), log_q_prior = get_noise_scale_proposal(noise, scale, params)
    K_XX = new_scale * forest_gram_matrix(forest, train_x, train_x, feat_types)
    K_XX_s = K_XX + (1e-6 + new_noise) * np.eye(K_XX.shape[0])
    new_K_inv = np.linalg.inv(K_XX_s)
    _, new_K_logdet = np.linalg.slogdet(K_XX_s)

    new_mll = mll(new_K_inv, new_K_logdet, train_y)
    log_ll = new_mll - cur_mll
    log_alpha = log_q_prior + log_ll

    if np.log(np.random.uniform()) <= min(log_alpha, 0):
        # accept - set the new mll and K_inv values
        cur_K_inv = new_K_inv
        cur_K_logdet = new_K_logdet
        cur_mll = new_mll
        scale = new_scale
        noise = new_noise

    return forest, noise, scale, cur_K_inv, cur_K_logdet, cur_mll
