from ...tree_models.tree_kernels import AlfalfaGP
from .transitions import Transition
from .tree_traversal import assign_node_depth

def acceptance_probability(model: AlfalfaGP, transition: Transition):
    # P(INVERSE_METHOD) / P(METHOD) * ...
    assign_node_depth(model.tree)
    q_ratio = transition.log_q_ratio()

