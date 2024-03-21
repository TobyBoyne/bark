import gpytorch as gpy
import numpy as np
import scipy.stats as stats
import torch
from tqdm import tqdm

from ...forest import AlfalfaTree
from ...tree_kernels import AlfalfaGP
from ...utils.logger import MCMCLogger
from .data import Data
from .noise_scale_transitions import (
    noise_acceptance_probability,
    propose_positive_transition,
    scale_acceptance_probability,
)
from .params import BARTTrainParams
from .quick_inverse import QuickInverter
from .tree_transitions import (
    propose_transition,
    tree_acceptance_probability,
)


def default_noise_prior():
    return stats.halfnorm(scale=10.0)


def default_scale_prior():
    return stats.halfnorm(scale=10.0)


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
        # create mll for logging only
        mll = gpy.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        with torch.no_grad():
            for _ in tqdm(range(self.params.warmup_steps)):
                self.step()

            for i in tqdm(range(self.params.n_steps)):
                self.step()
                if i % self.params.lag == 0:
                    self.logger.checkpoint(self.model)
                    output = self.model(self.model.train_inputs[0])
                    likelihood = -mll(output, self.model.train_targets)
                    self.logger.log(mll=likelihood)

        return self.logger

    def step(self):
        if isinstance(self.model.tree_model, AlfalfaTree):
            self._transition_tree(self.model.tree_model)
        else:
            quick_inverter = QuickInverter(self.model)
            for tree in self.model.tree_model.trees:
                self._transition_tree(tree, quick_inverter)

        self._transition_noise()
        self._transition_scale()

    def _accept_transition(self, log_alpha):
        return np.log(np.random.rand()) <= log_alpha

    def _transition_tree(self, tree: AlfalfaTree, quick_inverter: QuickInverter):
        transition = propose_transition(self.data, tree, self.params)
        if transition is None:
            # not a valid transition
            return

        log_alpha = tree_acceptance_probability(
            self.data, self.model, transition, self.params, quick_inverter
        )
        if self._accept_transition(log_alpha):
            transition.apply()
            quick_inverter.cache()

    def _transition_noise(self):
        new_noise = propose_positive_transition(self.model.likelihood.noise.item())
        log_alpha = noise_acceptance_probability(
            self.model, new_noise, self.noise_prior
        )
        if self._accept_transition(log_alpha):
            self.logger.log(accept_noise=True)
            self.model.likelihood.noise = new_noise
        else:
            self.logger.log(accept_noise=False)

    def _transition_scale(self):
        new_scale = propose_positive_transition(
            self.model.covar_module.outputscale.item()
        )
        log_alpha = scale_acceptance_probability(
            self.model, new_scale, self.scale_prior
        )
        if self._accept_transition(log_alpha):
            self.model.covar_module.outputscale = new_scale
