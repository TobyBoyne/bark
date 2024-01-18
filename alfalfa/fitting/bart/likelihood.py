from ...tree_models.tree_kernels import AlfalfaGP
from .transitions import Transition

def acceptance_probability(model: AlfalfaGP, transition: Transition):
    q_ratio = transition.log_q_ratio()
    
