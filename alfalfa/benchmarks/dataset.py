"""Real datasets

Many obtained from https://archive.ics.uci.edu/"""

import numpy as np
from jaxtyping import Float, Shaped
from ucimlrepo import fetch_ucirepo

from ..utils.space import (
    CategoricalDimension,
    ContinuousDimension,
    IntegerDimension,
    Space,
)
from .base import DatasetFunc


class UCIDatasetFunc(DatasetFunc, skip_validation=True):
    dataset_name: str
    target_feature: str

    def _load_data(self) -> tuple[Shaped[np.ndarray, "N D"], Float[np.ndarray, "N"]]:
        dataset = fetch_ucirepo(name=self.dataset_name)
        nan_idxs = dataset.data.features.isna().any(axis=1)
        X = dataset.data.features[~nan_idxs].to_numpy()
        y = dataset.data.targets[self.target_feature][~nan_idxs].to_numpy().reshape(-1)

        return X, y


class AutoMPG(UCIDatasetFunc):
    dataset_name = "Auto MPG"
    target_feature = "mpg"

    @property
    def space(self):
        return Space(
            [
                ContinuousDimension(key="displacement", bnds=[0.0, 500.0]),
                IntegerDimension(key="cylinders", bnds=[3, 8]),
                ContinuousDimension(key="horsepower", bnds=[0.0, 500.0]),
                ContinuousDimension(key="weight", bnds=[0.0, 7000.0]),
                ContinuousDimension(key="acceleration", bnds=[0.0, 30.0]),
                IntegerDimension(key="model_year", bnds=[70, 82]),
                IntegerDimension(key="origin", bnds=[1, 3]),
            ]
        )


class StudentPerformance(UCIDatasetFunc):
    dataset_name = "Student Performance"
    target_feature = "G3"

    @property
    def space(self):
        return Space(
            [
                CategoricalDimension(key="school", bnds=["GP", "MS"]),
                CategoricalDimension(key="sex", bnds=["M", "F"]),
                IntegerDimension(key="age", bnds=[15, 22]),
                CategoricalDimension(key="address", bnds=["U", "R"]),
                CategoricalDimension(key="famsize", bnds=["LE3", "GT3"]),
                CategoricalDimension(key="Pstatus", bnds=["A", "T"]),
                IntegerDimension(key="Medu", bnds=[0, 4]),
                IntegerDimension(key="Fedu", bnds=[0, 4]),
                CategoricalDimension(
                    key="Mjob",
                    bnds=["teacher", "health", "services", "at_home", "other"],
                ),
                CategoricalDimension(
                    key="Fjob",
                    bnds=["teacher", "health", "services", "at_home", "other"],
                ),
                CategoricalDimension(
                    key="reason", bnds=["home", "reputation", "course", "other"]
                ),
                CategoricalDimension(
                    key="guardian", bnds=["mother", "father", "other"]
                ),
                IntegerDimension(key="traveltime", bnds=[1, 4]),
                IntegerDimension(key="studytime", bnds=[1, 4]),
                IntegerDimension(key="failures", bnds=[0, 4]),
                CategoricalDimension(key="schoolsup", bnds=["yes", "no"]),
                CategoricalDimension(key="famsup", bnds=["yes", "no"]),
                CategoricalDimension(key="paid", bnds=["yes", "no"]),
                CategoricalDimension(key="activities", bnds=["yes", "no"]),
                CategoricalDimension(key="nursery", bnds=["yes", "no"]),
                CategoricalDimension(key="higher", bnds=["yes", "no"]),
                CategoricalDimension(key="internet", bnds=["yes", "no"]),
                CategoricalDimension(key="romantic", bnds=["yes", "no"]),
                IntegerDimension(key="famrel", bnds=[1, 5]),
                IntegerDimension(key="freetime", bnds=[1, 5]),
                IntegerDimension(key="goout", bnds=[1, 5]),
                IntegerDimension(key="Dalc", bnds=[1, 5]),
                IntegerDimension(key="Walc", bnds=[1, 5]),
                IntegerDimension(key="health", bnds=[1, 5]),
                IntegerDimension(key="absences", bnds=[0, 93]),
            ]
        )


class Abalone(UCIDatasetFunc):
    dataset_name = "Abalone"
    target_feature = "Rings"

    @property
    def space(self):
        return Space(
            [
                CategoricalDimension(key="Sex", bnds=["M", "F", "I"]),
                ContinuousDimension(key="Length", bnds=[0.0, 1.0]),
                ContinuousDimension(key="Diameter", bnds=[0.0, 1.0]),
                ContinuousDimension(key="Height", bnds=[0.0, 2.0]),
                ContinuousDimension(key="Whole_weight", bnds=[0.0, 3.0]),
                ContinuousDimension(key="Shucked_weight", bnds=[0.0, 1.5]),
                ContinuousDimension(key="Viscera_weight", bnds=[0.0, 1.0]),
                ContinuousDimension(key="Shell_weight", bnds=[0.0, 2.0]),
            ]
        )


class ConcreteCompressive(UCIDatasetFunc):
    dataset_name = "Concrete Compressive Strength"
    target_feature = "Concrete compressive strength"

    @property
    def space(self):
        return Space(
            [
                ContinuousDimension(key="Cement", bnds=[0.0, 600.0]),
                ContinuousDimension(key="Blast Furnace Slag", bnds=[0.0, 400.0]),
                ContinuousDimension(key="Fly Ash", bnds=[0.0, 210.0]),
                ContinuousDimension(key="Water", bnds=[0.0, 250.0]),
                ContinuousDimension(key="Superplasticizer", bnds=[0.0, 50.0]),
                ContinuousDimension(key="Coarse Aggregate", bnds=[0.0, 1200.0]),
                ContinuousDimension(key="Fine Aggregate", bnds=[0.0, 1000.0]),
                ContinuousDimension(key="Age", bnds=[0.0, 400.0]),
            ]
        )
