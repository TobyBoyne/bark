from .params import BARTTrainParams
from .transitions import propose_transition
from .likelihood import acceptance_probability
from ...tree_models.tree_kernels import AlfalfaGP
from .data import Data
import torch
import numpy as np

class BART:
    def __init__(self, model: AlfalfaGP, data: Data, params: BARTTrainParams):
        self.model = model
        self.data = data
        self.params = params

    def run(self):
        with torch.no_grad():
            for _ in range(100):
                self.step()

    def step(self):
        tree = self.model.tree_model
        # these two methods should probably be a part of the BART class
        transition = propose_transition(self.data, tree, self.params)
        if transition is None:
            # not a valid transition
            return
        
        alpha = acceptance_probability(self.data, self.model, transition, self.params)
        u = np.log(np.random.rand())
        if u <= alpha:
            # accept transition
            transition.apply()

        print(tree.root)
               
