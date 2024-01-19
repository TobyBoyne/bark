from ...tree_models.tree_kernels import AlfalfaGP
from .transitions import Transition
from .tree_traversal import assign_node_depth
from .data import Data
from .params import BARTTrainParams

def acceptance_probability(data: Data, model: AlfalfaGP, transition: Transition, params: BARTTrainParams):
    # P(INVERSE_METHOD) / P(METHOD) * ...
    assign_node_depth(model.tree)
    q_ratio = transition.log_q_ratio(data)
    likelihood_ratio = transition.log_likelihood_ratio(model)
    prior_ratio = transition.log_prior_ratio(data, params.alpha, params.beta)

    return min(q_ratio + likelihood_ratio + prior_ratio, 0.0)