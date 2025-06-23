import gurobipy
import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MinimizeObjective
from pandas import DataFrame

from bark.bofire_utils.constraints import FunctionalInequalityConstraint
from bark.bofire_utils.domain import build_integer_input


class DiscreteAckley(Benchmark):
    """
    adapted from: https://arxiv.org/pdf/2210.10199"""

    def __init__(self, discrete_dim=10, cont_dim=3, **kwargs):
        super().__init__(**kwargs)
        self.dim = discrete_dim + cont_dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        build_integer_input(key=f"x_{i}", bounds=(0, 1))
                        for i in range(discrete_dim)
                    ),
                    *(
                        ContinuousInput(key=f"x_{i + discrete_dim}", bounds=(-1.0, 1.0))
                        for i in range(cont_dim)
                    ),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: DataFrame) -> DataFrame:
        x_int = X[self.domain.inputs.get_keys(includes=DiscreteInput)].to_numpy()
        x_cont = X[self.domain.inputs.get_keys(includes=ContinuousInput)].to_numpy()
        # map x_int from {0, 1} to {-1, 1}
        x_int = 2 * x_int - 1
        z = np.concatenate([x_int, x_cont], axis=1)
        a = 20.0
        b = 0.2
        c = 2 * np.pi
        d = self.dim
        y = (
            -a * np.exp(-b * np.sqrt(1 / d * np.sum(z**2, axis=1)))
            - np.exp(1 / d * np.sum(np.cos(c * z), axis=1))
            + a
            + np.exp(1)
        )
        return pd.DataFrame(data=y[:, None], columns=self.domain.outputs.get_keys())


class DiscreteRosenbrock(Benchmark):
    """Adapted from: https://arxiv.org/pdf/2210.10199"""

    def __init__(self, discrete_dim=6, cont_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = discrete_dim + cont_dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        build_integer_input(key=f"x_{i}", bounds=(-1, 2))
                        for i in range(discrete_dim)
                    ),
                    *(
                        ContinuousInput(
                            key=f"x_{i + discrete_dim}", bounds=(-5.0, 10.0)
                        )
                        for i in range(cont_dim)
                    ),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: DataFrame) -> DataFrame:
        x_int = X[self.domain.inputs.get_keys(includes=DiscreteInput)].to_numpy()
        x_cont = X[self.domain.inputs.get_keys(includes=ContinuousInput)].to_numpy()
        # map x_int from [-1, 2] to [-5, 10]
        x_int = 5 * x_int
        z = np.concatenate([x_int, x_cont], axis=1)
        y = np.sum(
            100 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1 - z[:, :-1]) ** 2, axis=1
        )
        return pd.DataFrame(data=y[:, None], columns=self.domain.outputs.get_keys())


class PressureVessel(Benchmark):
    # adapted from: https://www.scielo.br/j/lajss/a/ZsdRkGWRVtDdHJP8WTDFFpB/?format=pdf&lang=en
    # note that the order of the features is based on sorted(inputs)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def _pv_func(x, model_core: gurobipy.Model | None = None):
            if model_core is not None:
                x2_squ = model_core.addVar(lb=10.0**2, ub=200.0**2)
                model_core.addConstr(x2_squ == x[0] * x[0])
            else:
                x2_squ = x[0] ** 2

            return -np.pi * x[1] * x2_squ - (4 / 3) * np.pi * x[0] * x2_squ

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
                    # LinearInequalityConstraint(features=["x_1", "x_2"], coefficients=[0.0625, 0.00954], rhs=0.0),
                    FunctionalInequalityConstraint(
                        func=lambda x, mc=None: -x[2] * 0.0625 + x[0] * 0.0193, rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, mc=None: -x[3] * 0.0625 + x[0] * 0.00954, rhs=0.0
                    ),
                    FunctionalInequalityConstraint(func=_pv_func, rhs=-1_296_000),
                ]
            ),
        )

    def _f(self, X: DataFrame) -> DataFrame:
        y = (
            0.6224 * (0.0625 * X["x_0"]) * X["x_2"] * X["x_3"]
            + 1.7781 * (0.0625 * X["x_1"]) * X["x_2"] ** 2
            + 3.1661 * X["x_3"] * (0.0625 * X["x_0"]) ** 2
            + 19.84 * X["x_2"] * (0.0625 * X["x_0"]) ** 2
        )
        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())

    def get_optima(self) -> DataFrame:
        return pd.DataFrame(
            data=[[13, 7, 42.09127, 176.7466, 6061.0778]],
            columns=["x_0", "x_1", "x_2", "x_3"] + self.domain.outputs.get_keys(),
        )


class CombinationFunc2(Benchmark):
    """Linear combination of 3 synthetic functions - Rosenbrock, 6-Hump Camel,
    and Beale

    Adapted from: https://arxiv.org/pdf/1906.08878
    Bayesian Optimisation over Multiple Continuous and Categorical Inputs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(
                        key="func_0", categories=["ros", "cam", "bea"]
                    ),  # ["ros", "cam", "bea"]
                    CategoricalInput(
                        key="func_1", categories=["ros", "cam", "bea"]
                    ),  # ["ros", "cam", "bea"]
                    ContinuousInput(key="x_0", bounds=(-1.0, 1.0)),
                    ContinuousInput(key="x_1", bounds=(-1.0, 1.0)),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

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

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys(includes=ContinuousInput)].to_numpy()
        functions = pd.DataFrame(
            data=np.transpose([self._rosenbrock(x), self._camel(x), self._beale(x)]),
            columns=["ros", "cam", "bea"],
        )

        idx, cols = pd.factorize(X["func_0"])
        f0 = functions.reindex(cols, axis=1).to_numpy()[np.arange(len(functions)), idx]

        idx, cols = pd.factorize(X["func_1"])
        f1 = functions.reindex(cols, axis=1).to_numpy()[np.arange(len(functions)), idx]

        y = (f0 + f1)[:, None]
        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())

    # def __call__(self, x, **kwargs):

    #     funcs = [
    #         self._rosenbrock,
    #         self._camel,
    #         self._beale,
    #     ]
    #     cont_x = np.atleast_2d(x)[:, 2:].astype(float)
    #     f1 = funcs[int(x[0])]
    #     f2 = funcs[int(x[1])]
    #     return (f1(cont_x) + f2(cont_x)).item()

    def get_optima(self) -> DataFrame:
        # y= -1.0316 * 2
        raise NotImplementedError()
