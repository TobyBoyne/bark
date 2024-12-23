import logging

import numpy as np
import pandas as pd
from bofire.data_models.features.api import (
    AnyFeature,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)
from bofire.strategies.api import PredictiveStrategy
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer

from bark.bofire_utils.data_models.strategies.smac import SMACStrategy as DataModel
from bark.bofire_utils.domain import get_feature_bounds

logging.getLogger(__name__)

try:  # noqa: E402
    from smac import HyperparameterOptimizationFacade, Scenario
    from smac.runhistory.dataclasses import TrialInfo, TrialValue
except ImportError:
    logging.warning("smac not installed; cannot use SMAC strategy")


def _bofire_feat_to_smac(feat: AnyFeature):
    if isinstance(feat, ContinuousInput):
        return Float(name=feat.key, bounds=get_feature_bounds(feat))
    if isinstance(feat, DiscreteInput):
        return Integer(name=feat.key, bounds=get_feature_bounds(feat))
    if isinstance(feat, CategoricalInput):
        return Categorical(name=feat.key, items=feat.categories)
    raise TypeError(f"Cannot convert feature of type {feat.type} to SMAC")


class SMACStrategy(PredictiveStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.configspace = ConfigurationSpace(seed=self.seed)
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
            lambda x: 0,  # dummy objective function
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

    def _tell(self, experiments: pd.DataFrame, **kwargs):
        for i, experiment in experiments.iterrows():
            experiment_dict = experiment.to_dict()
            cfg = Configuration(self.configspace, values=experiment_dict)
            trial = TrialInfo(cfg, seed=self.seed)
            value = TrialValue(experiment_dict[self.domain.outputs.get_keys()[0]])
            self.smac.tell(trial, value)

    def _ask(self, candidate_count: int) -> np.ndarray:
        assert candidate_count == 1, "SMAC only supports single candidates"
        if self.smac is None:
            raise ValueError("SMAC not initialized")
        config = self.smac.ask()
        return _postprocess_candidate(config)

    def _postprocess_candidate(self, config: Configuration) -> pd.DataFrane:
        return pd.DataFrame(config)
