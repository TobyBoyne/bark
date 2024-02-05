import numpy as np
import scipy.stats as stats
import torch
from tqdm import tqdm

from alfalfa.utils.logger import MCMCLogger

from ...forest import AlfalfaTree
from ...tree_kernels import AlfalfaGP
from .data import Data
from .noise_scale_transitions import (noise_acceptance_probability,
                                      propose_noise_transition,
                                      propose_scale_transition,
                                      scale_acceptance_probability)
from .params import BARTTrainParams
from .tree_transitions import propose_transition, tree_acceptance_probability


def default_noise_prior():
    return stats.gamma(a=1.0)


def default_scale_prior():
    return stats.halfnorm(scale=1.0)


class BART:
    def __init__(
        self,
        model: AlfalfaGP,
        data: Data,
        params: BARTTrainParams,
        noise_prior=None,
        scale_prior=None,
    ):
        self.model = model
        self.data = data
        self.params = params

        self.logger = MCMCLogger()
        self.noise_prior = default_noise_prior() if noise_prior is None else noise_prior
        self.scale_prior = default_scale_prior() if scale_prior is None else scale_prior

    def run(self):
        with torch.no_grad():
            for _ in tqdm(range(self.params.warmup_steps)):
                self.step()

            for i in tqdm(range(self.params.n_steps)):
                self.step()
                if i % self.params.lag == 0:
                    self.logger.checkpoint(self.model)

        return self.logger

    def step(self):
        if isinstance(self.model.tree_model, AlfalfaTree):
            self._transition_tree(self.model.tree_model)
        else:
            for tree in self.model.tree_model.trees:
                self._transition_tree(tree)

        self._transition_noise()
        self._transition_scale()
        self.logger.log(noise=self.model.likelihood.noise)
        self.logger.log(scale=self.model.covar_module.outputscale)

    def _accept_transition(self, log_alpha):
        return np.log(np.random.rand()) <= log_alpha

    def _transition_tree(self, tree: AlfalfaTree):
        # these two methods should probably be a part of the BART class
        transition = propose_transition(self.data, tree, self.params)
        if transition is None:
            # not a valid transition
            return

        log_alpha = tree_acceptance_probability(
            self.data, self.model, transition, self.params
        )
        if self._accept_transition(log_alpha):
            transition.apply()

    def _transition_noise(self):
        new_noise = propose_noise_transition(self.model)
        log_alpha = noise_acceptance_probability(
            self.model, new_noise, self.noise_prior
        )
        if self._accept_transition(log_alpha):
            self.model.likelihood.noise = new_noise

    def _transition_scale(self):
        new_scale = propose_scale_transition(self.model)
        log_alpha = scale_acceptance_probability(
            self.model, new_scale, self.scale_prior
        )
        if self._accept_transition(log_alpha):
            self.model.covar_module.outputscale = new_scale
