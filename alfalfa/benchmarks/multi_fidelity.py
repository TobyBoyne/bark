import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput, TaskInput


class CurrinExp2D(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x_0", bounds=(0.0, 1.0)),
                    ContinuousInput(key="x_1", bounds=(0.0, 1.0)),
                    TaskInput(key="task", categories=["ground_truth", "local_avg"]),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
        )

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.domain.inputs.get_keys(includes=ContinuousInput)].to_numpy()
        i = X["task"].to_numpy().reshape(1, -1)

        # the low fidelity function is a local average of high fidelity
        deltas = 0.05 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        x_tilde = x[..., None, :] + deltas

        s = np.mean(self._evaluate(np.clip(x_tilde, 0, 1)), axis=-1)
        y = np.where(i == 0, self._evaluate(x), s)

        return pd.DataFrame(data=y, columns=self.domain.outputs.get_keys())

    # def __call__(self, x, i, **kwargs):
    #     x = np.asarray(x).reshape(-1, 2)
    #     i = np.asarray(i).reshape(-1)

    #     # The low fidelity function is a local average of high fidelity
    #     deltas = 0.05 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    #     x_tilde = x[..., None, :] + deltas
    #     s = np.mean(self._evaluate(np.clip(x_tilde, 0, 1)), axis=-1)

    #     y = np.where(i == 0, self._evaluate(x), s)
    #     return y

    def _evaluate(self, x: np.ndarray):
        x0 = x[..., 0]
        x1 = x[..., 1]
        prod1 = 1 - np.exp(-1 / (2 * (x1 + 1e-5)))
        prod2 = (2300 * x0**3 + 1900 * x0**2 + 2092 * x0 + 60) / (
            100 * x0**3 + 500 * x0**2 + 4 * x0 + 20
        )

        return -prod1 * prod2 / 10
