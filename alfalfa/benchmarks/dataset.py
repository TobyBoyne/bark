"""Real datasets

Many obtained from https://archive.ics.uci.edu/"""

import numpy as np
from jaxtyping import Float, Shaped
from ucimlrepo import fetch_ucirepo

from ..utils.space import (
    ContinuousDimension,
    IntegerDimension,
    Space,
)
from .base import DatasetFunc


class UCIDatasetFunc(DatasetFunc, skip_validation=True):
    dataset_name: str

    def _load_data(self) -> tuple[Shaped[np.ndarray, "N D"], Float[np.ndarray, "N"]]:
        dataset = fetch_ucirepo(name=self.dataset_name)
        nan_idxs = dataset.data.features.isna().any(axis=1)
        X = dataset.data.features[~nan_idxs].to_numpy()
        y = dataset.data.targets[~nan_idxs].to_numpy().reshape(-1)

        return X, y


class AutoMPG(UCIDatasetFunc):
    dataset_name = "Auto MPG"

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
