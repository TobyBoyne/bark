"""https://github.com/huawei-noah/HEBO/blob/master/MCBO/mcbo/tasks/xgboost_opt/xgboost_opt_task.py"""

import xgboost
from sklearn import datasets, metrics, model_selection

from alfalfa.benchmarks.base import SynFunc


class XGBoostMNIST(SynFunc):
    int_idx = {0, 1, 2, 4}

    def __init__(self, seed: int, split=6 / 7):
        super().__init__(seed)
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

    def __call__(self, x, **kwargs):
        xgboost_kwargs = {
            "booster": ["gbtree", "dart"][int(x[0])],
            "grow_policy": ["depthwise", "lossguide"][int(x[1])],
            "objective": ["multi:softmax", "multi:softprob"][int(x[2])],
            "learning_rate": 10 ** x[3],
            "max_depth": int(x[4]),
            "min_split_loss": x[5],
            "subsample": x[6],
            "reg_lambda": x[7],
        }
        model = xgboost.XGBClassifier(**xgboost_kwargs)
        model.fit(self.train_x, self.train_y)

        y_pred = model.predict(self.test_x)

        # 1-acc for minimization
        score = 1 - metrics.accuracy_score(self.test_y, y_pred)

        return score

    @property
    def bounds(self):
        return [
            [0, 1],  # booster
            [0, 1],  # grow_policy
            [0, 1],  # objective
            [-5, 0],  # log learning_rate
            [1, 10],  # max_depth
            [0, 10],  # min_split_loss
            [0.001, 1],  # subsample
            [0, 5],  # reg_lambda
        ]
