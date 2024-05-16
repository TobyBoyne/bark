# adapted from https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/synthetic/pest.py

import numpy as np

from ..utils.space import CategoricalDimension, Space
from .base import SynFunc


def spread_pests(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    else:
        init_pest_frac = np.random.beta(
            init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
        )
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        else:
            spread_rate = np.random.beta(
                spread_alpha, spread_beta, size=(n_simulations,)
            )
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            else:
                control_rate = np.random.beta(
                    control_alpha, control_beta[x[i]], size=(n_simulations,)
                )
            next_pest_frac = spread_pests(
                curr_pest_frac, spread_rate, control_rate, True
            )
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                1.0
                - control_price_max_discount[x[i]]
                / float(n_stages)
                * float(np.sum(x == x[i]))
            )
        else:
            next_pest_frac = spread_pests(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class PestControl(SynFunc):
    """
    Pest Control Problem.
    """

    categories = [
        "do nothing",
        "pesticide 1",
        "pesticide 2",
        "pesticide 3",
        "pesticide 4",
    ]

    def __init__(self, seed: int, n_stages: int = 25):
        super().__init__(seed)
        self._n_stages = n_stages

    def __call__(self, x: list | np.ndarray) -> float:
        # x_ = x.replace(['do nothing', 'pesticide 1', 'pesticide 2', 'pesticide 3', 'pesticide 4'], [0, 1, 2, 3, 4])
        x = np.asarray(x)
        if isinstance(x[0], str):
            x = self.space.transform(x)
        assert x.ndim == 1 and len(x) == self._n_stages, (x.shape, self._n_stages)
        evaluation = _pest_control_score(x, seed=self.seed)
        return evaluation

    @property
    def space(self) -> Space:
        dims = [
            CategoricalDimension(key=f"stage_{i}", bnds=PestControl.categories)
            for i in range(self._n_stages)
        ]
        return Space(dims)

    @property
    def optimum(self) -> float:
        # TODO: confirm this value
        return 11.5
