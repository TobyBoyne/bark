from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class LeafGPSurrogate(Surrogate, TrainableSurrogate):
    pass


class BARKSurrogate(Surrogate, TrainableSurrogate):
    # MCMC run parameters
    warmup_steps: int = 50
    num_samples: int = 5
    steps_per_sample: int = 10

    # node depth prior
    alpha: float = 0.95
    beta: float = 2.0
    num_trees: int = 50

    # transition type probabilities
    grow_prune_weight: float = 0.5
    change_weight: float = 1.0

    num_chains: int = 1
    verbose: bool = False
