import gurobipy
import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective
from pandas import DataFrame

from alfalfa.bofire_utils.constraints import FunctionalInequalityConstraint
from alfalfa.utils.domain import build_integer_input


def _pv_func(x: list[float | gurobipy.Var], model_core: gurobipy.Model | None = None):
    if model_core is not None:
        x2_squ = model_core.addVar(lb=10.0**2, ub=200.0**2)
        model_core.addConstr(x2_squ == x[2] * x[2])
    else:
        x2_squ = x[2] ** 2

    return -np.pi * x[3] * x2_squ - (4 / 3) * np.pi * x[2] * x2_squ


class PressureVessel(Benchmark):
    # adapted from: https://www.scielo.br/j/lajss/a/ZsdRkGWRVtDdHJP8WTDFFpB/?format=pdf&lang=en
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    build_integer_input(key="x_0", bounds=(1, 99)),
                    build_integer_input(key="x_1", bounds=(1, 99)),
                    ContinuousInput(
                        key="x_2",
                        bounds=(10.0, 200.0),
                    ),
                    ContinuousInput(
                        key="x_3",
                        bounds=(10.0, 200.0),
                    ),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    # LinearInequalityConstraint(features=["x_0", "x_2"], coefficients=[0.0625, 0.0193], rhs=0.0),
                    # LinearInequalityConstraint(features=["x_1", "x_3"], coefficients=[0.0625, 0.00954], rhs=0.0),
                    FunctionalInequalityConstraint(
                        func=lambda x, mc: x[0] * 0.0625 + x[2] * 0.0193, rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, mc: x[1] * 0.0625 + x[3] * 0.00954, rhs=0.0
                    ),
                    FunctionalInequalityConstraint(func=_pv_func, rhs=-1_296_000),
                ]
            ),
        )

    def _f(self, X: DataFrame) -> DataFrame:
        return (
            0.6224 * X["x_0"] * X["x_2"] * X["x_3"]
            + 1.7781 * (0.0625 * X["x_1"]) * X["x_2"] ** 2
            + 3.1661 * X["x_3"] * (0.0625 * X["x_0"]) ** 2
            + 19.84 * X["x_2"] * (0.0625 * X["x_0"]) ** 2
        )

    def get_optima(self) -> DataFrame:
        return pd.DataFrame(
            data=[13, 7, 42.09127, 176.7466, 6061.0778],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )
