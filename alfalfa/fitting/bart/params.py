from dataclasses import dataclass
import numpy as np
from enum import Enum, auto

class TransitionEnum(Enum):
    GROW = 0
    PRUNE = 1

@dataclass
class BARTTrainParams:
    alpha: float = 0.95
    beta: float = 0.5

    grow_weight: float = 0.5
    prune_weight: float = 0.5

    @property
    def step_weights(self):
        p = np.array([self.grow_weight, self.prune_weight])
        return p / np.sum(p)