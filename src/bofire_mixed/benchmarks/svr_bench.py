import logging
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from bofire_mixed.domain import build_integer_input

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SVRBench(Benchmark):
    def __init__(
        self,
        num_features_to_optimize: int = 50,
        num_features_to_keep: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="log_epsilon", bounds=(-2, 0)),
                    ContinuousInput(key="log_C", bounds=(-2, 2)),
                    ContinuousInput(key="log_gamma", bounds=(-1, 1)),
                    *[
                        build_integer_input(key=f"feature_{i + 1}", bounds=[0, 1])
                        for i in range(num_features_to_optimize)
                    ],
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )
        self.num_features_to_optimize = num_features_to_optimize
        self.num_features_to_keep = num_features_to_keep
        self.evaluations = 0
        self.x_trains, self.x_tests, self.y_trains, self.y_tests = self.prepare_x_y()
        logging.info(
            f"Optimizing {self.num_features_to_optimize} features from {self.num_features_to_keep} features"
        )

    def prepare_x_y(
        self, num_samples: int = 10000
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        if getattr(self, "x_trains", None) is not None:
            return self.x_trains, self.x_tests, self.y_trains, self.y_tests

        dataset_path = (
            pathlib.Path(__file__).parent / "data" / "slice_localization_data.csv"
        )
        data = pd.read_csv(dataset_path, sep=",").to_numpy()

        x = data[:, :-1]
        y = data[:, -1]

        # remove constant features
        features_to_keep = (x.max(0) - x.min(0)) > 1e-6
        x = x[:, features_to_keep]

        mixed_inds = np.random.RandomState(0).permutation(len(x))

        x = x[mixed_inds[:num_samples]]
        y = y[mixed_inds[:num_samples]]

        # select most important features using XGBoost
        feature_select_regr = XGBRegressor(max_depth=8).fit(x, y)
        feature_select_inds = np.argsort(feature_select_regr.feature_importances_)[
            ::-1
        ][: self.num_features_to_keep]  # Keep 50 features
        x = x[:, feature_select_inds]

        # create train and test splits
        x_trains, y_trains, x_tests, y_tests = [], [], [], []

        for seed in range(5):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
                x, y, test_size=0.3, random_state=seed
            )
            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)
        return x_trains, x_tests, y_trains, y_tests

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Evaluating SVR ({self.evaluations})")
        self.x_trains, self.x_tests, self.y_trains, self.y_tests = self.prepare_x_y()

        evaluations = []
        for i in range(len(X)):
            logging.info(f"Evaluating SVR ({self.evaluations}) - {i}")
            svm_hyp = X.iloc[i]

            scores = []

            for j in range(5):
                x_train, x_test, y_train, y_test = (
                    self.x_trains[j],
                    self.x_tests[j],
                    self.y_trains[j],
                    self.y_tests[j],
                )

                # standardize y_train
                y_train_mean, y_train_std = y_train.mean(), y_train.std()
                y_train = (y_train - y_train_mean) / y_train_std

                # select features
                features_filter = np.array(
                    [
                        int(getattr(svm_hyp, f"feature_{j + 1}"))
                        for j in range(self.num_features_to_optimize)
                    ]
                    + [1] * (self.num_features_to_keep - self.num_features_to_optimize)
                ).astype(int)

                if np.sum(features_filter) == 0:  # nothing selected
                    y_pred = y_train_mean * np.ones(len(x_test))

                else:
                    x_train = x_train[:, features_filter]
                    x_test = x_test[:, features_filter]
                    learner = SVR(
                        epsilon=np.exp(svm_hyp.log_epsilon),
                        C=np.exp(svm_hyp.log_C),
                        gamma=np.exp(svm_hyp.log_gamma) / x_train.shape[-1],
                    )
                    regr = make_pipeline(MinMaxScaler(), learner)
                    regr.fit(x_train, y_train)
                    y_pred = regr.predict(x_test) * y_train_std + y_train_mean
                scores.append(mean_squared_error(y_test, y_pred))
            evaluations.append(np.mean(scores))
        self.evaluations += 1
        return pd.DataFrame(
            data=np.array(evaluations).reshape(-1, 1),
            columns=self.domain.outputs.get_keys(),
        )
