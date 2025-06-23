import logging

import numpy as np
import pandas as pd
from bofire.data_models.features.api import (
    AnyFeature,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.api import PredictiveStrategy

from bofire_mixed.data_models.strategies.smac import SMACStrategy as DataModel
from bofire_mixed.domain import get_feature_bounds

logging.getLogger(__name__)

try:  # noqa: E402
    import ConfigSpace as cs
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.runhistory.dataclasses import TrialInfo, TrialValue
except ImportError:
    logging.warning("smac not installed; cannot use SMAC strategy")


def _bofire_feat_to_smac(feat: AnyFeature):
    if isinstance(feat, ContinuousInput):
        return cs.Float(name=feat.key, bounds=get_feature_bounds(feat))
    if isinstance(feat, DiscreteInput):
        return cs.Integer(name=feat.key, bounds=get_feature_bounds(feat))
    if isinstance(feat, CategoricalInput):
        return cs.Categorical(name=feat.key, items=feat.categories)
    raise TypeError(f"Cannot convert feature of type {feat.type} to SMAC")


class SMACStrategy(PredictiveStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.configspace = cs.ConfigurationSpace(seed=self.seed)
        self.smac: HyperparameterOptimizationFacade | None = None
        self._init_smac()

    def _init_smac(self):
        for feat in self.domain.inputs.get():
            self.configspace.add(_bofire_feat_to_smac(feat))

        scenario = Scenario(self.configspace, deterministic=True, n_trials=100)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1,  # use one seed per config
        )

        initial_design = HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=1
        )

        self.smac = HyperparameterOptimizationFacade(
            scenario,
            lambda x, seed=0: 0,  # dummy objective function
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

    def _fit(self, experiments: pd.DataFrame, **kwargs):
        for i, experiment in experiments.iterrows():
            experiment_dict = experiment[self.domain.inputs.get_keys()].to_dict()
            cfg = cs.Configuration(self.configspace, values=experiment_dict)
            trial = TrialInfo(cfg, seed=self.seed)
            value = TrialValue(experiment[self.domain.outputs.get_keys()].item())
            if trial not in self.smac.runhistory:
                self.smac.tell(trial, value, save=False)

    def _ask(self, candidate_count: int) -> np.ndarray:
        assert candidate_count == 1, "SMAC only supports single candidates"
        if self.smac is None:
            raise ValueError("SMAC not initialized")
        trial_info = self.smac.ask()
        return self._postprocess_candidate(trial_info.config)

    def _postprocess_candidate(self, config: "cs.Configuration") -> pd.DataFrame:
        df_candidate = pd.DataFrame(dict(config), index=(0,))
        preds = self.predict(df_candidate)
        return pd.concat((df_candidate, preds), axis=1)

    def _predict(self, experiments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros((experiments.shape[0], 1))
        return zeros, zeros

    def has_sufficient_experiments(self):
        return len(self.experiments) >= 1

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        return {}
