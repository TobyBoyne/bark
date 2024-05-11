from dataclasses import dataclass
from enum import Enum

import numpy as np


class TransitionEnum(Enum):
    GROW = 0
    PRUNE = 1
    CHANGE = 2


@dataclass
class BARTTrainParams:
    # MCMC run parameters
    warmup_steps: int = 50
    n_steps: int = 50
    lag: int = 5

    # node depth prior
    alpha: float = 0.95
    beta: float = 2

    # transition type probabilities
    grow_prune_weight: float = 0.5
    change_weight: float = 1.0

    @property
    def step_weights(self):
        p = np.array(
            [self.grow_prune_weight, self.grow_prune_weight, self.change_weight]
        )
        return p / np.sum(p)
