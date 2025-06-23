import gurobipy
import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective

from bark.bofire_utils.constraints import (
    FunctionalEqualityConstraint,
    FunctionalInequalityConstraint,
)


class G1(Benchmark):
    # adapted from: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page506.htm
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key=f"x_{i}", bounds=(0.0, 100.0 if i in {9, 10, 11} else 1.0)
                    )
                    for i in range(13)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    LinearInequalityConstraint(
                        features=["x_0", "x_1", "x_9", "x_10"],
                        coefficients=[2, 2, 1, 1],
                        rhs=10,
                    ),
                    LinearInequalityConstraint(
                        features=["x_0", "x_2", "x_9", "x_11"],
                        coefficients=[2, 2, 1, 1],
                        rhs=10,
                    ),
                    LinearInequalityConstraint(
                        features=["x_1", "x_2", "x_10", "x_11"],
                        coefficients=[2, 2, 1, 1],
                        rhs=10,
                    ),
                    LinearInequalityConstraint(
                        features=["x_0", "x_9"], coefficients=[-8, 1], rhs=0
                    ),
                    LinearInequalityConstraint(
                        features=["x_1", "x_10"], coefficients=[-8, 1], rhs=0
                    ),
                    LinearInequalityConstraint(
                        features=["x_2", "x_11"], coefficients=[-3, 1], rhs=0
                    ),
                    LinearInequalityConstraint(
                        features=["x_3", "x_4", "x_9"], coefficients=[-2, -1, 1], rhs=0
                    ),
                    LinearInequalityConstraint(
                        features=["x_5", "x_6", "x_10"], coefficients=[-2, -1, 1], rhs=0
                    ),
                    LinearInequalityConstraint(
                        features=["x_7", "x_8", "x_11"], coefficients=[-2, -1, 1], rhs=0
                    ),
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        f = (
            5 * np.sum(x[:, :4], axis=1)
            - 5 * np.sum(x[:, :4] ** 2, axis=1)
            - np.sum(x[:, 4:], axis=1)
        )[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())

    def get_optima(self):
        x = [3 if i in {9, 10, 11} else 1 for i in range(13)]
        return pd.DataFrame(
            data=[x + [-15]],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class G3(Benchmark):
    # adapted from: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2613.htm
    def __init__(self, dim: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(0.0, 1.0))
                    for i in range(self.dim)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    FunctionalEqualityConstraint(
                        func=lambda x, _=None: sum(
                            [x[i] * x[i] for i in range(self.dim)]
                        ),
                        rhs=1.0,
                    )
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        z = np.sqrt(self.dim) ** self.dim
        f = z * np.prod(x, axis=1)[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())

    def get_optima(self):
        x = [1 / np.sqrt(self.dim) for _ in range(self.dim)]
        return pd.DataFrame(
            data=[x + [1.0]],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class G4(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def u(x):
            return (
                85.334407
                + 0.0056858 * x[1] * x[4]
                + 0.0006262 * x[0] * x[3]
                - 0.0022053 * x[2] * x[4]
            )

        def v(x):
            return (
                80.51249
                + 0.0071317 * x[1] * x[4]
                + 0.0029955 * x[0] * x[1]
                + 0.0021813 * x[2] ** 2
            )

        def w(x):
            return (
                9.300961
                + 0.0047026 * x[2] * x[4]
                + 0.0012547 * x[0] * x[2]
                + 0.0019085 * x[2] * x[3]
            )

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x_0", bounds=(78.0, 102.0)),
                    ContinuousInput(key="x_1", bounds=(33.0, 45.0)),
                    ContinuousInput(key="x_2", bounds=(27.0, 45.0)),
                    ContinuousInput(key="x_3", bounds=(27.0, 45.0)),
                    ContinuousInput(key="x_4", bounds=(27.0, 45.0)),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: -u(x), rhs=0.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: u(x), rhs=92.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: -v(x), rhs=-90.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: v(x), rhs=110.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: -w(x), rhs=-20.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: w(x), rhs=25.0
                    ),
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        f = (
            5.3578547 * x[:, 2] ** 2
            + 0.8356891 * x[:, 0] * x[:, 4]
            + 37.293239 * x[:, 0]
            - 40792.141
        )[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())

    def get_optima(self):
        x = [78, 33, 29.995, 45, 36.7758]
        y = -30665.539
        return pd.DataFrame(
            data=[x + [y]],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class G6(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x_0", bounds=(13.0, 100.0)),
                    ContinuousInput(key="x_1", bounds=(0.0, 100.0)),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: -((x[0] - 5) ** 2) - (x[1] - 5) ** 2,
                        rhs=-100.0,
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: (x[0] - 6) ** 2 + (x[1] - 5) ** 2,
                        rhs=82.81,
                    ),
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        f = ((x[:, 0] - 10) ** 3 + (x[:, 1] - 20) ** 3)[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())


class G7(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def g4(x, _=None):
            return 3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3]

        def g5(x, _=None):
            return 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3]

        def g6(x, _=None):
            return 0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 - x[5]

        def g7(x, _=None):
            return (
                x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
            )

        def g8(x, _=None):
            return -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(-10.0, 10.0))
                    for i in range(10)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    LinearInequalityConstraint(
                        features=["x_0", "x_1", "x_6", "x_7"],
                        coefficients=[4, 5, -3, 9],
                        rhs=105.0,
                    ),
                    LinearInequalityConstraint(
                        features=["x_0", "x_1", "x_6", "x_7"],
                        coefficients=[10, -8, -17, 2],
                        rhs=0.0,
                    ),
                    LinearInequalityConstraint(
                        features=["x_0", "x_1", "x_8", "x_9"],
                        coefficients=[-8, 2, 5, -2],
                        rhs=12.0,
                    ),
                    FunctionalInequalityConstraint(func=g4, rhs=120.0),
                    FunctionalInequalityConstraint(func=g5, rhs=40.0),
                    FunctionalInequalityConstraint(func=g6, rhs=30.0),
                    FunctionalInequalityConstraint(func=g7, rhs=0.0),
                    FunctionalInequalityConstraint(func=g8, rhs=0.0),
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        f = (
            x[:, 0] ** 2
            + x[:, 1] ** 2
            + x[:, 0] * x[:, 1]
            - 14 * x[:, 0]
            - 16 * x[:, 1]
            + (x[:, 2] - 10) ** 2
            + 4 * (x[:, 3] - 5) ** 2
            + (x[:, 4] - 3) ** 2
            + 2 * (x[:, 5] - 1) ** 2
            + 5 * x[:, 6] ** 2
            + 7 * (x[:, 7] - 11) ** 2
            + 2 * (x[:, 8] - 10) ** 2
            + (x[:, 9] - 7) ** 2
            + 45
        )[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())

    def get_optima(self):
        # 24.3062091
        raise NotImplementedError()


class G10(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lb = [100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        ub = [10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(lb[i], ub[i]))
                    for i in range(8)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    LinearInequalityConstraint(
                        features=["x_3", "x_5"], coefficients=[1.0, 1.0], rhs=400.0
                    ),
                    LinearInequalityConstraint(
                        features=["x_3", "x_4", "x_6"],
                        coefficients=[-1.0, 1.0, 1.0],
                        rhs=400.0,
                    ),
                    LinearInequalityConstraint(
                        features=["x_4", "x_7"], coefficients=[-1.0, 1.0], rhs=100.0
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: 100 * x[0]
                        - x[0] * x[5]
                        + 833.33252 * x[3],
                        rhs=83333.333,
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: x[1] * x[3]
                        - x[1] * x[6]
                        - 1250 * x[3]
                        + 1250 * x[4],
                        rhs=0.0,
                    ),
                    FunctionalInequalityConstraint(
                        func=lambda x, _=None: x[2] * x[4] - x[2] * x[7] - 2500 * x[4],
                        rhs=-1250000,
                    ),
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        f = (x[:, 0] + x[:, 1] + x[:, 2])[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())

    def get_optima(self):
        x = [
            579.3167,
            1359.943,
            5110.071,
            182.0174,
            295.5985,
            217.9799,
            286.4162,
            395.5979,
        ]
        return pd.DataFrame(
            data=[x + [7049.3307]],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class Alkylation(Benchmark):
    # original source: R. N. Sauer, A. R. Colville and C. W. Bunvick,
    #                  ‘Computer points the way to more profits’, Hydrocarbon Process.
    #                  Petrol. Refiner. 43,8492 (1964).
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/cons.py

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lb = [0.1, 0.0, 0.0, 0.0, 90.0, 0.01, 145.0]
        ub = [2000.0, 16000.0, 120.0, 5000.0, 95.0, 4.0, 162.0]

        def x5(x, model_core=None):
            if model_core is None:
                return 1.22 * x[3] - x[0]
            return model_core.getVarByName("_helper_x5")

        def x6(x, model_core=None):
            if model_core is None:
                return (98000.0 * x[2]) / (x[3] * x[5] + 1000.0 * x[2])
            return model_core.getVarByName("_helper_x6")

        def x8(x, model_core=None):
            if model_core is None:
                x5_ = x5(x)
                return (x[1] + x5_) / x[0]
            return model_core.getVarByName("_helper_x8")

        def x8_sq(x, model_core=None):
            if model_core is None:
                return x8(x) ** 2
            return model_core.getVarByName("_helper_x8_sq")

        def _setup_aux_vars(x, model_core: gurobipy.Model | None = None):
            if model_core is None:
                return
            # x = model_core._cont_var_dict

            # add x5 constr
            x5 = model_core.addVar(name="_helper_x5", lb=0.0, ub=2000.0)
            model_core.addConstr(x5 == 1.22 * x[3] - x[0])

            # add x6 constrs
            x35 = model_core.addVar(
                name="_helper_x35", lb=lb[3] * lb[5], ub=ub[3] * ub[5]
            )
            model_core.addConstr(x35 == x[3] * x[5])

            x6 = model_core.addVar(name="_helper_x6", lb=85.0, ub=93.0)
            model_core.addConstr(x6 * x35 + 1000.0 * x[2] * x6 == 98000.0 * x[2])

            # add x8 constrs
            x8 = model_core.addVar(name="_helper_x8", lb=3.0, ub=12.0)
            model_core.addConstr(x8 * x[0] == x[1] + x5)

            squ_x8 = model_core.addVar(name="_helper_x8_sq", lb=3.0**2, ub=12.0**2)
            model_core.addConstr(squ_x8 == x8 * x8)
            model_core.update()

        def g1(x, model_core=None):
            _setup_aux_vars(x, model_core)
            x8_ = x8(x, model_core)
            x8_sq_ = x8_sq(x, model_core)
            return 0.99 * x[3] - (x[0] * (1.12 + 0.13167 * x8_ - 0.00667 * x8_sq_))

        def g2(x, model_core=None):
            x8_ = x8(x, model_core)
            x8_sq_ = x8_sq(x, model_core)
            return (x[0] * (1.12 + 0.13167 * x8_ - 0.00667 * x8_sq_)) - (
                100.0 / 99.0
            ) * x[3]

        def g3(x, model_core=None):
            x6_ = x6(x, model_core)
            x8_ = x8(x, model_core)
            x8_sq_ = x8_sq(x, model_core)
            return 0.99 * x[4] - (
                86.35 + 1.098 * x8_ - 0.038 * x8_sq_ + 0.325 * (x6_ - 89.0)
            )

        def g4(x, model_core=None):
            x6_ = x6(x, model_core)
            x8_ = x8(x, model_core)
            x8_sq_ = x8_sq(x, model_core)
            return (86.35 + 1.098 * x8_ - 0.038 * x8_sq_ + 0.325 * (x6_ - 89.0)) - (
                100.0 / 99.0
            ) * x[4]

        def g5(x, _=None):
            return 0.9 * x[5] - (35.82 - 0.222 * x[6])

        def g6(x, _=None):
            return (35.82 - 0.222 * x[6]) - (10.0 / 9.0) * x[5]

        def g7(x, _=None):
            return 0.99 * x[6] - (-133 + 3 * x[4])

        def g8(x, _=None):
            return (-133 + 3.0 * x[4]) - (100.0 / 99.0) * x[6]

        def g9(x, model_core=None):
            x5_ = x5(x, model_core)
            return x5_ - 2000

        def g10(x, model_core=None):
            x5_ = x5(x, model_core)
            return -x5_

        def g11(x, model_core=None):
            x6_ = x6(x, model_core)
            return x6_ - 93.0

        def g12(x, model_core=None):
            x6_ = x6(x, model_core)
            return 85.0 - x6_

        def g13(x, model_core=None):
            x8_ = x8(x, model_core)
            return x8_ - 12.0

        def g14(x, model_core=None):
            x8_ = x8(x, model_core)
            return 3.0 - x8_

        # TODO: confirm g9-g14 are necessary
        con_funcs = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14]

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(lb[i], ub[i]))
                    for i in range(7)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    FunctionalInequalityConstraint(func=func, rhs=0.0)
                    for func in con_funcs
                ]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        x5 = 1.22 * x[:, 3] - x[:, 0]
        f = -(
            0.063 * x[:, 3] * x[:, 4]
            - 5.04 * x[:, 0]
            - 0.035 * x[:, 1]
            - 10.0 * x[:, 2]
            - 3.36 * x5
        )[:, None]
        return pd.DataFrame(data=f, columns=self.domain.outputs.get_keys())

    def get_optima(self):
        x = [1698.1, 15819, 54.107, 3031.2, 95.000, 1.5618, 153.54]
        return pd.DataFrame(
            data=[x + [-1768.75]],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )
