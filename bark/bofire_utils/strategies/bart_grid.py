import numpy as np
import pandas as pd
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.api import PredictiveStrategy, RandomStrategy
from bofire.utils.naming_conventions import (
    get_column_names,
)

from bark.bofire_utils.data_models.strategies.bart_grid import (
    BARTGridStrategy as BARTGridStrategyDataModel,
)
from bark.bofire_utils.data_models.surrogates.mapper import surrogate_map
from bark.bofire_utils.surrogates.bart import BARTSurrogate


class BARTGridStrategy(PredictiveStrategy):
    def __init__(self, data_model: BARTGridStrategyDataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)

        self.surrogate_specs = data_model.surrogate_specs
        self.bart_surrogate: BARTSurrogate = surrogate_map(
            data_model=data_model.surrogate_specs
        )

        self.sampler = RandomStrategy(
            data_model=RandomStrategyDataModel(
                domain=self.domain,
                seed=self.seed,
                fallback_sampling_method=SamplingMethodEnum.SOBOL,
            )
        )

    def _fit(self, experiments: pd.DataFrame, **kwargs):
        self.bart_surrogate.fit(experiments)

    def _predict(self, experiments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.bart_surrogate._predict(experiments)

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        assert candidate_count == 1, "BART only supports single candidates"
        D = len(self.domain.inputs)
        n = min(2 ** (5 * D), 2**14)
        sobol_grid = self.sampler.ask(n)
        samples = self.bart_surrogate.function_samples(sobol_grid)
        # samples has shape (chain, draw, n)
        # ucb = E_y [ max(mu - sqrt(kappa * pi / 2) * abs(y - mu)) ]
        # Wilson et al 2018 (Eq 7) for minimization
        y_pred = samples.posterior_predictive.y_pred.to_numpy().reshape(-1, n)
        mu = y_pred.mean(axis=0).reshape(1, -1)

        kappa = 1.96
        reparam = -mu + kappa * np.sqrt(np.pi / 2) * np.abs(y_pred - mu)
        lcb = np.mean(reparam, axis=0)
        idx = np.argmax(lcb)
        candidate_df = sobol_grid.iloc[idx : idx + 1]
        return self._postprocess_candidate(candidate_df)

    def _postprocess_candidate(self, candidate: pd.DataFrame) -> pd.DataFrame:
        candidate = self.domain.inputs.inverse_transform(
            candidate, self.input_preprocessing_specs
        ).reset_index(drop=True)

        # this is too expensive
        preds = self.predict(candidate)
        pred_col, sd_col = get_column_names(self.domain.outputs)
        preds = pd.DataFrame(
            data=[[0.0, 1.0]],
            columns=pred_col + sd_col,
        )
        return pd.concat((candidate, preds), axis=1)

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
