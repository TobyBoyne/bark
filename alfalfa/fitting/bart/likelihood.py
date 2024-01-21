from ...tree_models.tree_kernels import AlfalfaGP
from .transitions import Transition
from .data import Data
from .params import BARTTrainParams
import numpy as np

def acceptance_probability(data: Data, model: AlfalfaGP, transition: Transition, params: BARTTrainParams):
    # TODO: P(INVERSE_METHOD) / P(METHOD) * ...
    q_ratio = transition.log_q_ratio(data)
    if np.isneginf(q_ratio):
        # no valid ways to perform the operation
        # e.g. there are no valid splitting rules for a given node
        return -np.inf
    likelihood_ratio = transition.log_likelihood_ratio(model)
    prior_ratio = transition.log_prior_ratio(data, params.alpha, params.beta)

    return min(q_ratio + likelihood_ratio + prior_ratio, 0.0)