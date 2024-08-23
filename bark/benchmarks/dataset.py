"""Real datasets

Many obtained from https://archive.ics.uci.edu/"""

import numpy as np
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from ucimlrepo import fetch_ucirepo

from bark.bofire_utils.domain import build_integer_input


def get_ucirepo_domain_and_data(
    dataset_name: str,
) -> tuple[Domain, np.ndarray, np.ndarray]:
    domain = _dataset_name_to_domain[dataset_name]
    dataset = fetch_ucirepo(name=dataset_name)
    target_feature = domain.outputs.get_keys()[0]
    nan_idxs = dataset.data.features.isna().any(axis=1)
    train_x = dataset.data.features[~nan_idxs].to_numpy()
    train_y = dataset.data.targets[target_feature][~nan_idxs].to_numpy().reshape(-1)
    return domain, (train_x, train_y)


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
