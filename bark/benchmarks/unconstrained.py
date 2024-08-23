import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective


class Friedman(Benchmark):
    """

    Multivariate adaptive regression splines (1999)"""

    def __init__(self, dim=10, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        ContinuousInput(key=f"x_{i}", bounds=(0.0, 1.0))
                        for i in range(dim)
                    )
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        y = (
            10 * np.sin(np.pi * x[:, 0] * x[:, 1])
            + 20 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3]
            + 5 * x[:, 4]
        )[:, None]
        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())


class Rastrigin(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, dim: int = 10, **kwargs):
        super().__init__(kwargs)
        self.dim = dim

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        ContinuousInput(key=f"x_{i}", bounds=(-4.0, 5.0))
                        for i in range(dim)
                    )
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        d = self.dim
        f = x**2 - 10.0 * np.cos(2.0 * np.pi * x)
        y = np.sum(f, axis=1) + 10.0 * d
        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())


class StyblinskiTang(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, dim: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        ContinuousInput(key=f"x_{i}", bounds=(-5.0, 5.0))
                        for i in range(dim)
                    )
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        y = 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x, axis=1)
        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())

    def get_optima(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=[([-2.903534] * self.dim) + [-39.16616 * self.dim]],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class Schwefel(Benchmark):
    # adapted from: https://github.com/solab-ntu/opt-prob-collect/blob/master/opt_prob/non_cons.py
    def __init__(self, dim: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        ContinuousInput(key=f"x_{i}", bounds=(-500.0, 500.0))
                        for i in range(dim)
                    )
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys()].to_numpy()
        f = np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
        y = 418.9829 * self.dim - f
        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())
