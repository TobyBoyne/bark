import logging
from typing import Tuple

import numpy as np
import pandas as pd
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDM
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.api import RandomStrategy
from bofire.strategies.predictives.predictive import PredictiveStrategy

from bark.optimizer.opt_core import get_opt_core_from_domain
from bark.optimizer.opt_model import build_opt_model_from_forest
from bark.optimizer.proposals import propose
from bofire_mixed.data_models.strategies.tree_kernel import (
    TreeKernelStrategy as DataModel,
)
from bofire_mixed.data_models.surrogates.mapper import surrogate_map

logger = logging.getLogger(__name__)


class TreeKernelStrategy(PredictiveStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)

        self.surrogate_specs = data_model.surrogate_specs
        self.tree_surrogate = surrogate_map(data_model=self.surrogate_specs)

        self.model_core = get_opt_core_from_domain(self.domain)

    def _fit(self, experiments: pd.DataFrame, **kwargs):
        self.tree_surrogate.fit(experiments)

    def _predict(self, experiments: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.tree_surrogate._predict(experiments)

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

        try:
            candidate = propose(self.domain, opt_model, self.model_core)
        except ValueError:
            logger.warning("Failed to optimize acqf, proposing random candidate.")
            random_strategy_dm = RandomStrategyDM(domain=self.domain, seed=self.seed)
            random_strategy = RandomStrategy(random_strategy_dm)
            candidate_df = random_strategy.ask(1)[self.domain.inputs.get_keys()]
            candidate = candidate_df.iloc[0].to_list()
        return self._postprocess_candidate(candidate)

    def _postprocess_candidate(self, candidate: list) -> pd.DataFrame:
        df_candidate = pd.DataFrame(
            data=[candidate], columns=self.domain.inputs.get_keys()
        )

        df_candidate = self.domain.inputs.inverse_transform(
            df_candidate, self.input_preprocessing_specs
        )

        preds = self.predict(df_candidate)
        return pd.concat((df_candidate, preds), axis=1)

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        return self.surrogate_specs.input_preprocessing_specs

    def has_sufficient_experiments(
        self,
    ) -> bool:
        if self.experiments is None:
            return False
        if (
            len(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    experiments=self.experiments,
                ),
            )
            > 1
        ):
            return True
        return False
