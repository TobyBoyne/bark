import numpy as np


class Standardize:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def __call__(self, y: np.ndarray, train: bool) -> np.ndarray:
        if train:
            self.mean = y.mean()
            self.std = max(y.std(), 1e-6)
        return (y - self.mean) / self.std

    def untransform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

    def untransform_mu_var(
        self, mu: np.ndarray, var: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.untransform(mu), var * self.std**2
