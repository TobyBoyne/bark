"""Real datasets

Many obtained from https://archive.ics.uci.edu/"""

import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from ucimlrepo import fetch_ucirepo

from bofire_mixed.domain import build_integer_input


def get_ucirepo_domain_and_data(
    dataset_name: str,
) -> tuple[Domain, pd.DataFrame]:
    domain = _dataset_name_to_domain[dataset_name]
    dataset = fetch_ucirepo(name=dataset_name)
    target_features = domain.outputs.get_keys()
    nan_idxs = dataset.data.features.isna().any(axis=1)
    train_x = dataset.data.features[~nan_idxs]
    train_y = dataset.data.targets[target_features][~nan_idxs]
    experiments = pd.concat((train_x, train_y), axis=1)
    return domain, experiments


_auto_mpg = Domain.from_lists(
    inputs=[
        ContinuousInput(key="displacement", bounds=[0.0, 500.0]),
        build_integer_input(key="cylinders", bounds=[3, 8]),
        ContinuousInput(key="horsepower", bounds=[0.0, 500.0]),
        ContinuousInput(key="weight", bounds=[0.0, 7000.0]),
        ContinuousInput(key="acceleration", bounds=[0.0, 30.0]),
        build_integer_input(key="model_year", bounds=[70, 82]),
        build_integer_input(key="origin", bounds=[1, 3]),
    ],
    outputs=[
        ContinuousOutput(key="mpg"),
    ],
)

_student_performance = Domain.from_lists(
    inputs=[
        CategoricalInput(key="school", categories=["GP", "MS"]),
        CategoricalInput(key="sex", categories=["M", "F"]),
        build_integer_input(key="age", bounds=[15, 22]),
        CategoricalInput(key="address", categories=["U", "R"]),
        CategoricalInput(key="famsize", categories=["LE3", "GT3"]),
        CategoricalInput(key="Pstatus", categories=["A", "T"]),
        build_integer_input(key="Medu", bounds=[0, 4]),
        build_integer_input(key="Fedu", bounds=[0, 4]),
        CategoricalInput(
            key="Mjob", categories=["teacher", "health", "services", "at_home", "other"]
        ),
        CategoricalInput(
            key="Fjob", categories=["teacher", "health", "services", "at_home", "other"]
        ),
        CategoricalInput(
            key="reason", categories=["home", "reputation", "course", "other"]
        ),
        CategoricalInput(key="guardian", categories=["mother", "father", "other"]),
        build_integer_input(key="traveltime", bounds=[1, 4]),
        build_integer_input(key="studytime", bounds=[1, 4]),
        build_integer_input(key="failures", bounds=[0, 4]),
        CategoricalInput(key="schoolsup", categories=["yes", "no"]),
        CategoricalInput(key="famsup", categories=["yes", "no"]),
        CategoricalInput(key="paid", categories=["yes", "no"]),
        CategoricalInput(key="activities", categories=["yes", "no"]),
        CategoricalInput(key="nursery", categories=["yes", "no"]),
        CategoricalInput(key="higher", categories=["yes", "no"]),
        CategoricalInput(key="internet", categories=["yes", "no"]),
        CategoricalInput(key="romantic", categories=["yes", "no"]),
        build_integer_input(key="famrel", bounds=[1, 5]),
        build_integer_input(key="freetime", bounds=[1, 5]),
        build_integer_input(key="goout", bounds=[1, 5]),
        build_integer_input(key="Dalc", bounds=[1, 5]),
        build_integer_input(key="Walc", bounds=[1, 5]),
        build_integer_input(key="health", bounds=[1, 5]),
        build_integer_input(key="absences", bounds=[0, 93]),
    ],
    outputs=[
        ContinuousOutput(key="G3"),
    ],
)

_abalone = Domain.from_lists(
    inputs=[
        CategoricalInput(key="Sex", categories=["M", "F", "I"]),
        ContinuousInput(key="Length", bounds=[0.0, 1.0]),
        ContinuousInput(key="Diameter", bounds=[0.0, 1.0]),
        ContinuousInput(key="Height", bounds=[0.0, 2.0]),
        ContinuousInput(key="Whole_weight", bounds=[0.0, 3.0]),
        ContinuousInput(key="Shucked_weight", bounds=[0.0, 1.5]),
        ContinuousInput(key="Viscera_weight", bounds=[0.0, 1.0]),
        ContinuousInput(key="Shell_weight", bounds=[0.0, 2.0]),
    ],
    outputs=[
        ContinuousOutput(key="Rings"),
    ],
)

_concrete_compressive = Domain.from_lists(
    inputs=[
        ContinuousInput(key="Cement", bounds=[0.0, 600.0]),
        ContinuousInput(key="Blast Furnace Slag", bounds=[0.0, 400.0]),
        ContinuousInput(key="Fly Ash", bounds=[0.0, 210.0]),
        ContinuousInput(key="Water", bounds=[0.0, 250.0]),
        ContinuousInput(key="Superplasticizer", bounds=[0.0, 50.0]),
        ContinuousInput(key="Coarse Aggregate", bounds=[0.0, 1200.0]),
        ContinuousInput(key="Fine Aggregate", bounds=[0.0, 1000.0]),
        ContinuousInput(key="Age", bounds=[0.0, 400.0]),
    ],
    outputs=[
        ContinuousOutput(key="Concrete compressive strength"),
    ],
)

_dataset_name_to_domain = {
    "Auto MPG": _auto_mpg,
    "Student Performance": _student_performance,
    "Abalone": _abalone,
    "Concrete Compressive Strength": _concrete_compressive,
}


class DatasetBenchmark(Benchmark):
    """Benchmark for regression datasets"""

    def __init__(self, dataset_name: str, standardise=True, **kwargs):
        domain, data = get_ucirepo_domain_and_data(dataset_name)

        self._domain = domain
        self.data = data
        if standardise:
            output_cols = self.domain.outputs.get_keys()
            y = self.data[output_cols]
            self.data[output_cols] = (y - y.mean()) / y.std()
        self._num_sampled = 0
        super().__init__(**kwargs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        idxs = candidates.index
        return self.data.loc[idxs][self.domain.outputs.get_keys()]

    def sample(self, n_samples: int, seed: int = 0) -> pd.DataFrame:
        assert self._num_sampled + n_samples <= len(self.data)
        data_order = np.random.default_rng(seed).permutation(len(self.data))
        sample_idxs = data_order[self._num_sampled : self._num_sampled + n_samples]
        samples = self.data.iloc[sample_idxs]
        self._num_sampled += n_samples

        return samples[self.domain.inputs.get_keys()]
