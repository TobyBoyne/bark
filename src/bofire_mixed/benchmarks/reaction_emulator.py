"""Taken from BoFire
(https://github.com/experimental-design/bofire/blob/main/tutorials/basic_examples/Reaction_Optimization_Example.ipynb)
"""

import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (  # we won't need all of those.
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)

# Reaction Optimization Notebook util code
T0 = 25
T1 = 100
e0 = np.exp((T1 + 0) / T0)
e60 = np.exp((T1 + 60) / T0)
de = e60 - e0

boiling_points = {  # in °C
    "MeOH": 64.7,
    "THF": 66.0,
    "Dioxane": 101.0,
}
density = {  # in kg/l
    "MeOH": 0.792,
    "THF": 0.886,
    "Dioxane": 1.03,
}
# create dict from individual dicts
descs = {
    "boiling_points": boiling_points,
    "density": density,
}
solvent_descriptors = pd.DataFrame(descs)


# these functions are for faking real experimental data ;)
def calc_volume_fact(V):
    # 20-90
    # max at 75 = 1
    # min at 20 = 0.7
    # at 90=0.5
    x = (V - 20) / 70
    x = 0.5 + (x - 0.75) * 0.1 + (x - 0.4) ** 2
    return x


def calc_rhofact(solvent_type, Tfact):
    #  between 0.7 and 1.1
    x = solvent_descriptors["density"][solvent_type]
    x = (1.5 - x) * (Tfact + 0.5) / 2
    return x.values


def calc_Tfact(T):
    x = np.exp((T1 + T) / T0)
    return (x - e0) / de


def evaluate(candidates: pd.DataFrame, nsamples=100, A=25, B=90):
    nsamples = len(candidates)
    T = candidates["Temperature"].values
    V = candidates["Solvent Volume"].values
    solvent_types = candidates["Solvent Type"].values

    Tfact = calc_Tfact(T)
    rhofact = calc_rhofact(solvent_types, Tfact)
    Vfact = calc_volume_fact(V)
    y = A * Tfact + B * rhofact
    y = 0.5 * y + 0.5 * y * Vfact
    samples = pd.DataFrame(
        {
            "Yield": -y,
            "valid_Yield": np.ones(nsamples),
        },
    )
    samples.index = pd.RangeIndex(nsamples)
    return samples


class ToyReaction(Benchmark):
    def __init__(self, **kwargs):
        temperature_feature = ContinuousInput(
            key="Temperature", bounds=[0.0, 60.0], unit="°C"
        )
        solvent_amount_feature = ContinuousInput(key="Solvent Volume", bounds=[20, 90])
        solvent_type_feature = CategoricalInput(
            key="Solvent Type",
            categories=["MeOH", "THF", "Dioxane"],
        )

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    temperature_feature,
                    solvent_type_feature,
                    solvent_amount_feature,
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="Yield", objective=MaximizeObjective())]
            ),
        )
        super().__init__(**kwargs)

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        return evaluate(X)
