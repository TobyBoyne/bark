# adapted from https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/synthetic/pest.py

import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective
from jaxtyping import Int
from pandas import DataFrame


def spread_pests(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x: Int[np.ndarray, "D"], rng: np.random.Generator):
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

    init_pest_frac = rng.beta(
        init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,)
    )
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        spread_rate = rng.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            control_rate = rng.beta(
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


class PestControl(Benchmark):
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

    def __init__(self, n_stages: int = 25, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._n_stages = n_stages
        self._pest_rng = np.random.default_rng(seed)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(key=f"stage_{i+1}", categories=self.categories)
                    for i in range(n_stages)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        stages = [f"stage_{i+1}" for i in range(self._n_stages)]
        specs = {s: CategoricalEncodingEnum.ORDINAL for s in stages}
        X_transformed = self.domain.inputs.transform(X, specs)
        X_numpy = X_transformed[stages].to_numpy()
        scores = []
        for row in X_numpy:
            score = _pest_control_score(row, self._pest_rng)
            scores.append([score])
        return pd.DataFrame(data=scores, columns=self.domain.outputs.get_keys())

    # def __call__(self, x: list | np.ndarray) -> float:
    #     # x_ = x.replace(['do nothing', 'pesticide 1', 'pesticide 2', 'pesticide 3', 'pesticide 4'], [0, 1, 2, 3, 4])
    #     x = np.asarray(x)
    #     if isinstance(x[0], str):
    #         x = self.space.transform(x)
    #     assert x.ndim == 1 and len(x) == self._n_stages, (x.shape, self._n_stages)
    #     evaluation = _pest_control_score(x, seed=self.seed)
    #     return evaluation

    def get_optima(self) -> DataFrame:
        # 11.5
        raise NotImplementedError()
