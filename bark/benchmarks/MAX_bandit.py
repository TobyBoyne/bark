import json
from pathlib import Path

import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective


class MAXBandit(Benchmark):
    def __init__(self, data_path: Path | str | None = None, **kwargs):
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    CategoricalInput(
                        key="A_ele",
                        categories=[
                            "Tl",
                            "Pb",
                            "Ge",
                            "Al",
                            "Ga",
                            "In",
                            "Sn",
                            "Cd",
                            "S",
                            "Si",
                            "As",
                            "P",
                        ],
                    ),
                    CategoricalInput(
                        key="M_ele",
                        categories=[
                            "Ti",
                            "V",
                            "Hf",
                            "Ta",
                            "Nb",
                            "Cr",
                            "Zr",
                            "Sc",
                            "Mo",
                        ],
                    ),
                    CategoricalInput(key="X_ele", categories=["N", "C"]),
                    ContinuousInput(key="e_a", bounds=[2.5, 6.0]),
                    ContinuousInput(key="APF", bounds=[0.35, 1.0]),
                    ContinuousInput(key="C", bounds=[-8.0, 0.0]),
                    ContinuousInput(key="m", bounds=[0.0, 1.0]),
                    ContinuousInput(key="Cv", bounds=[0.0, 1.0]),
                    ContinuousInput(key="a_exp", bounds=[2.0, 4.0]),
                    ContinuousInput(key="c_exp", bounds=[10.0, 25.0]),
                    ContinuousInput(key="Z", bounds=[10.0, 100.0]),
                    ContinuousInput(key="I_dist", bounds=[0.0, 1.0]),
                    *(
                        ContinuousInput(key=f"nuisance{i}", bounds=[-1.0, 1.0])
                        for i in range(16)
                    ),
                ]
            ),
            outputs=Outputs(
                # TODO: check that this is not a maximization problem
                # target is K_exp
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

        if data_path is None:
            data_path = Path(__file__).parent.joinpath("data/MAX_data.json")
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found at {data_path}")
        self.data = self._load_data_from_path(data_path)

    def _load_data_from_path(self, data_path: str):
        with open(data_path, "r") as f:
            data = json.load(f)

        return pd.DataFrame(
            data, columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys()
        )

    def _f(self, X: pd.DataFrame, **kwargs):
        idx = self.data.iloc[:, :-1].eq(X.iloc[0, :-1]).all(axis=1).idxmax()
        return self.data.iloc[idx][self.domain.outputs.get_keys()]

    def f_by_idx(self, idx: np.ndarray | int):
        return self.data.iloc[idx]

    def get_optima(self):
        opt_idx = self.data["y"].idxmin()
        return self.data.iloc[opt_idx]
