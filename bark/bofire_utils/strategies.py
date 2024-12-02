from typing import Tuple

import numpy as np
import pandas as pd
from bofire.strategies.predictives.predictive import PredictiveStrategy

from bark.bofire_utils.data_models.mapper import surrogate_map
from bark.bofire_utils.data_models.strategies import TreeKernelStrategy as DataModel
from bark.optimizer.opt_core import get_opt_core_from_domain
from bark.optimizer.opt_model import build_opt_model_from_forest
from bark.optimizer.proposals import propose


class TreeKernelStrategy(PredictiveStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)

        self.surrogate_specs = data_model.surrogate_specs
        self.tree_surrogate = surrogate_map(data_model=self.surrogate_specs)

        self.model_core = get_opt_core_from_domain(self.domain)

    def _fit(self, experiments: pd.DataFrame, **kwargs):
        self.tree_surrogate.fit(experiments)

    def _predict(self, experiments: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.tree_surrogate.predict(experiments)

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        assert candidate_count == 1, "BARK only supports single candidates"
        samples = self.tree_surrogate.model_as_tuple()
        data = self.tree_surrogate.train_data
        opt_model = build_opt_model_from_forest(
            domain=self.domain,
            gp_samples=samples,
            data=data,
            kappa=1.96,
            model_core=self.model_core,
        )

        next_x = propose(self.domain, opt_model, self.model_core)
        candidate = pd.DataFrame(data=[next_x], columns=self.domain.inputs.get_keys())
        return candidate
