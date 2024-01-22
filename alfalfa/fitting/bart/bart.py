from .params import BARTTrainParams
from .tree_transitions import propose_transition, tree_acceptance_probability
from .noise_scale_transitions import propose_noise_transition, noise_acceptance_probability
from ...tree_models.forest import AlfalfaTree
from ...tree_models.tree_kernels import AlfalfaGP
from .data import Data
import torch
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

class BART:
    def __init__(self, model: AlfalfaGP, data: Data, params: BARTTrainParams):
        self.model = model
        self.data = data
        self.params = params

    def run(self):
        with torch.no_grad():
            for _ in tqdm(range(self.params.warmup_steps)):
                self.step()

            for _ in tqdm(range(self.params.n_steps)):
                self.step()

    def step(self):
        if isinstance(self.model.tree_model, AlfalfaTree):
            self._transition_tree(tree)
        else:
            for tree in self.model.tree_model.trees:
                self._transition_tree(tree)

        self._transition_noise()
        self._transition_scale()

    def _accept_transition(self, log_alpha):
        return np.log(np.random.rand()) <= log_alpha

    def _transition_tree(self, tree: AlfalfaTree):
        # these two methods should probably be a part of the BART class
        transition = propose_transition(self.data, tree, self.params)
        if transition is None:
            # not a valid transition
            return
        
        log_alpha = tree_acceptance_probability(self.data, self.model, transition, self.params)
        if self._accept_transition(log_alpha):
            transition.apply()

            
    def _transition_noise(self):
        prior = stats.halfnorm(scale=1.0)
        new_noise = propose_noise_transition(self.model)
        log_alpha = noise_acceptance_probability(self.model, new_noise, prior)
        if self._accept_transition(log_alpha):
            self.model.likelihood.noise = new_noise

    def _transition_scale(self):
        prior = stats.halfnorm(scale=1.0)
        new_noise = propose_noise_transition(self.model)
        log_alpha = noise_acceptance_probability(self.model, new_noise, prior)
        if self._accept_transition(log_alpha):
            self.model.likelihood.noise = new_noise