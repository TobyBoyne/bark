from dataclasses import dataclass
import numpy as np
from enum import Enum, auto

class TransitionEnum(Enum):
    GROW = 0
    PRUNE = 1
    CHANGE = 2

@dataclass
class BARTTrainParams:
    warmup_steps: int = 50
    n_steps: int = 50
    alpha: float = 0.95
    beta: float = 0.5

    grow_prune_weight: float = 0.5
    change_weight: float = 1.0

    @property
    def step_weights(self):
        p = np.array([
            self.grow_prune_weight, 
            self.grow_prune_weight,
            self.change_weight
        ])
        return p / np.sum(p)