import numpy as np

from .base import SynFunc


class Himmelblau1D(SynFunc):
    def __call__(self, x, **kwargs):
        f = (x[0] ** 2 - 0.5) ** 2
        # output should be N(f_bar; 0, 1)
        f_bar = (f - 7 / 60) * np.sqrt(525 / 4)
        return f_bar * np.sin(10.0 * x[0])

    @property
    def bounds(self):
        return [[0.0, 1.0]]

    @property
    def optimum(self):
        # at x = 0.763
        return -1.2278


class Branin(SynFunc):
    # branin, rescaled to [0.0, 1.0]
    def __call__(self, x, **kwargs):
        x1 = x[0]
        x2 = x[1]
        x1_b = 15 * x1 - 5
        x2_b = 15 * x2
        A = (x2_b - (5.1 / (4 * np.pi**2)) * x1_b**2 + (5 / np.pi) * x1_b - 6) ** 2
        B = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1_b) - 44.81
        return -(A + B) / 51.95

    @property
    def bounds(self):
        return [[0.0, 1.0], [0.0, 1.0]]


class Friedman(SynFunc):
    """

    Multivariate adaptive regression splines (1999)"""

    is_vectorised = True

    def __init__(self, seed, dim=10):
        super().__init__(seed)
        self.dim = dim

    def __call__(self, x):
        return (
            10 * np.sin(np.pi * x[:, 0] * x[:, 1])
            + 20 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3]
            + 5 * x[:, 4]
        )

    @property
    def bounds(self):
        return [[0.0, 1.0] for _ in range(self.dim)]


class Hartmann6D(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py

    def __call__(self, x, **kwargs):
        a = np.asarray(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )

        c = np.asarray([1.0, 1.2, 3.0, 3.2])
        p = np.asarray(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )

        s = 0

        for i in range(1, 5):
            sm = 0
            for j in range(1, 7):
                sm = sm + a[i - 1, j - 1] * (x[j - 1] - p[i - 1, j - 1]) ** 2
            s = s + c[i - 1] * np.exp(-sm)

        y = -s
        return y

    @property
    def bounds(self):
        return [[0.0, 1.0] for _ in range(6)]

    @property
    def optimum(self):
        # at x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
        return -3.32237


class Rastrigin(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, seed, dim=10):
        super().__init__(seed)
        self.dim = dim

    def __call__(self, x, **kwargs):
        d = self.dim
        total = 0
        for xi in x:
            total = total + (xi**2 - 10.0 * np.cos(2.0 * np.pi * xi))
        f = 10.0 * d + total
        return f

    @property
    def bounds(self):
        return [[-4.0, 5.0] for _ in range(self.dim)]


class StyblinskiTang(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    is_vectorised = True

    def __init__(self, seed, dim=10):
        super().__init__(seed)
        self.dim = dim

    def __call__(self, x, **kwargs):
        x = np.atleast_2d(x)
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x, axis=1)

    @property
    def bounds(self):
        return [[-5.0, 5.0] for _ in range(self.dim)]

    @property
    def optimum(self):
        # at x = [-2.903534, ... , -2.903534]
        return -39.16616 * self.dim


class Schwefel(SynFunc):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, seed, dim=10):
        super().__init__(seed)
        self.dim = dim

    def __call__(self, x, **kwargs):
        d = self.dim
        total = 0
        for ii in range(d):
            xi = x[ii]
            total = total + xi * np.sin(np.sqrt(abs(xi)))
        f = 418.9829 * d - total
        return f

    @property
    def bounds(self):
        return [[-500.0, 500.0] for _ in range(self.dim)]


class CombinationFunc2(SynFunc):
    """Linear combination of 3 synthetic functions - Rosenbrock, 6-Hump Camel,
    and Beale

    Adapted from: https://arxiv.org/pdf/1906.08878
    Bayesian Optimisation over Multiple Continuous and Categorical Inputs
    """

    cat_idx = [0, 1]

    @property
    def bounds(self):
        return [
            [0, 1, 2],  # ["ros", "cam", "bea"],
            [0, 1, 2],  # ["ros", "cam", "bea"],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]

    def _rosenbrock(self, x: np.ndarray):
        return np.sum(
            100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1
        )

    def _camel(self, x: np.ndarray):
        return (
            4 * x[:, 0] ** 2
            - 2.1 * x[:, 0] ** 4
            + x[:, 0] ** 6 / 3
            + x[:, 0] * x[:, 1]
            - 4 * x[:, 1] ** 2
            + 4 * x[:, 1] ** 4
        )

    def _beale(self, x: np.ndarray):
        return (
            (1.5 - x[:, 0] + x[:, 0] * x[:, 1]) ** 2
            + (2.25 - x[:, 0] + x[:, 0] * x[:, 1] ** 2) ** 2
            + (2.625 - x[:, 0] + x[:, 0] * x[:, 1] ** 3) ** 2
        )

    def __call__(self, x, **kwargs):
        # funcs = {
        #     "ros": self._rosenbrock,
        #     "cam": self._camel,
        #     "bea": self._beale,
        # }
        funcs = [
            self._rosenbrock,
            self._camel,
            self._beale,
        ]
        cont_x = np.atleast_2d(x)[:, 2:].astype(float)
        f1 = funcs[int(x[0])]
        f2 = funcs[int(x[1])]
        return (f1(cont_x) + f2(cont_x)).item()

    @property
    def optimum(self):
        return -1.0316 * 2
