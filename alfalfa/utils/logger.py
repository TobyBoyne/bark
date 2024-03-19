"""Logging helpers for analysing bottlenecks - not a part of public API."""
from collections import defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np
from beartype.typing import Callable
from matplotlib.axes import Axes

from ..tree_kernels import AlfalfaGP


class Timer:
    """Context manager for timing function calls."""

    def __init__(self):
        self.current_key = None
        self.current_start = None
        self.durations = {}

    def __call__(self, key):
        self.current_key = key
        self.current_start = perf_counter()
        return self

    def __enter__(self):
        pass

    def __exit__(self, *args):
        num_calls, duration = self.durations.get(self.current_key, (0, 0.0))
        self.durations[self.current_key] = (
            num_calls + 1,
            perf_counter() - self.current_start + duration,
        )

    def average(self, unit=1.0):
        return {k: (v[1] / v[0]) * unit for k, v in self.durations.items()}

    def __getitem__(self, key):
        return self.durations[key]

    def __repr__(self):
        return str(self.durations)


class Logger:
    """Logger for recording loss functions over time."""

    def __init__(self):
        self.logs: dict[str, list] = defaultdict(list)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.logs[key].append(value)

    def __getitem__(self, key):
        return self.logs[key]


class BOLogger(Logger):
    """Logger for recording Bayesian Optimisation performance"""

    def __init__(self, initial_train_x, initial_train_y, target: Callable):
        self.num_initial = len(initial_train_x)
        self.target = target
        super().__init__()
        for x, y in zip(initial_train_x, initial_train_y):
            self.log(xs=x, ys=y)

    def log_bo_step(self, *, gp_mean, gp_confidence, new_train_x, new_train_y):
        super().log(
            gp_mean=gp_mean, gp_confidence=gp_confidence, xs=new_train_x, ys=new_train_y
        )

    @property
    def num_bo_steps(self):
        return len(self["gp_mean"])

    @property
    def running_best_eval(self):
        return np.minimum.accumulate(self["ys"])

    def plot_bo_step(self, ax: Axes, step_idx: int, test_x: np.ndarray):
        ylim = ax.get_ylim()
        ax.clear()
        i = step_idx + self.num_initial
        lower, upper = self["gp_confidence"][step_idx]

        scat = ax.scatter(self["xs"][:i], self["ys"][:i], marker="x", color="black")
        (line,) = ax.plot(
            test_x, self["gp_mean"][step_idx], color="b", label="GP Prediction"
        )
        fill = ax.fill_between(test_x, lower, upper, alpha=0.5, color="C0")
        (l_target,) = ax.plot(test_x, self.target(test_x), color="C1", label="Target")
        ax.set_ylim(ylim)
        ax.legend()
        return [scat, line, fill, l_target]


class MCMCLogger(Logger):
    def checkpoint(self, model: AlfalfaGP):
        self.log(samples=deepcopy(model.state_dict()))
        self.log(noise=model.likelihood.noise)
        self.log(scale=model.covar_module.outputscale)
