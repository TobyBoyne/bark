import numpy as np
from numba import njit


@njit
def get_noise_proposal(forest: np.ndarray, noise, scale, rng: np.random.Generator):
    pass


@njit
def get_scale_proposal(forest: np.ndarray, noise, scale, rng: np.random.Generator):
    pass
