import numpy as np

from .base import MFSynFunc


class CurrinExp2D(MFSynFunc):
    is_vectorised = True

    def __call__(self, x, i, **kwargs):
        x = np.asarray(x).reshape(-1, 2)
        i = np.asarray(i).reshape(-1)

        # The low fidelity function is a local average of high fidelity
        deltas = 0.05 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        x_tilde = x[..., None, :] + deltas
        s = np.mean(self._evaluate(np.clip(x_tilde, 0, 1)), axis=-1)

        y = np.where(i == 0, self._evaluate(x), s)
        return y

    @property
    def costs(self):
        return [10.0, 1.0]

    def _evaluate(self, x):
        x0 = x[..., 0]
        x1 = x[..., 1]
        prod1 = 1 - np.exp(-1 / (2 * (x1 + 1e-5)))
        prod2 = (2300 * x0**3 + 1900 * x0**2 + 2092 * x0 + 60) / (
            100 * x0**3 + 500 * x0**2 + 4 * x0 + 20
        )

        return -prod1 * prod2 / 10

    @property
    def bounds(self):
        return [[0.0, 1.0], [0.0, 1.0]]
