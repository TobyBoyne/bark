"""https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/xgboost_opt/xgboost_opt_task.py"""

import pandas as pd
import xgboost
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective
from sklearn import datasets, metrics, model_selection

from bark.bofire_utils.domain import build_integer_input


class XGBoostMNIST(Benchmark):
    def __init__(self, seed: int, split=0.3, **kwargs):
        super().__init__(**kwargs)
        data = datasets.load_digits()
        (
            self.train_x,
            self.test_x,
            self.train_y,
            self.test_y,
        ) = model_selection.train_test_split(
            data["data"],
            data["target"],
            test_size=split,
            stratify=data["target"],
            random_state=seed,
        )

        self._domain = Domain(
            inputs=Inputs(
                features=[
                    build_integer_input(key="booster", bounds=[0, 1]),
                    build_integer_input(key="grow_policy", bounds=[0, 1]),
                    build_integer_input(key="objective", bounds=[0, 1]),
                    ContinuousInput(key="log_learning_rate", bounds=[-5, 0]),
                    build_integer_input(key="max_depth", bounds=[1, 10]),
                    ContinuousInput(key="min_split_loss", bounds=[0, 10]),
                    ContinuousInput(key="subsample", bounds=[0.001, 1]),
                    ContinuousInput(key="reg_lambda", bounds=[0, 5]),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

    def _train_xgboost(self, xgboost_kwargs):
        model = xgboost.XGBClassifier(**xgboost_kwargs)
        model.fit(self.train_x, self.train_y)
        y_pred = model.predict(self.test_x)

        # 1-acc for minimization
        score = 1 - metrics.accuracy_score(self.test_y, y_pred)
        return score

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ys = []
        for _, row in X.iterrows():
            xgboost_kwargs = row.to_dict()
            log_learning_rate = xgboost_kwargs.pop("log_learning_rate")
            xgboost_kwargs["learning_rate"] = 10**log_learning_rate
            xgboost_kwargs["max_depth"] = int(xgboost_kwargs["max_depth"])
            xgboost_kwargs["booster"] = ["gbtree", "dart"][
                int(xgboost_kwargs.pop("booster"))
            ]
            xgboost_kwargs["grow_policy"] = [
                "depthwise",
                "lossguide",
            ][int(xgboost_kwargs.pop("grow_policy"))]
            xgboost_kwargs["objective"] = [
                "multi:softmax",
                "multi:softprob",
            ][int(xgboost_kwargs.pop("objective"))]

            ys.append([self._train_xgboost(xgboost_kwargs)])

        return pd.DataFrame(data=ys, columns=self.domain.outputs.get_keys())
