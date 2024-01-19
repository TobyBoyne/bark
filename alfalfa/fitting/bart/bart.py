from .params import BARTTrainParams
from .transitions import propose_transition
from .likelihood import acceptance_probability
from ...tree_models.tree_kernels import AlfalfaGP
from .data import Data
import torch

class BART:
    def __init__(self, model: AlfalfaGP, data: Data, params: BARTTrainParams):
        self.model = model
        self.data = data
        self.params = params

    def run(self):
        with torch.no_grad():
            self.step()

    def step(self):
        tree = self.model.tree
        # these two methods should probably be a part of the BART class
        transition = propose_transition(self.data, tree, self.params)
        alpha = acceptance_probability(self.data, self.model, transition, self.params)
        u = torch.log(torch.rand(()))
        if u <= alpha:
            # accept transition
            with transition:
                transition.accept()

               
