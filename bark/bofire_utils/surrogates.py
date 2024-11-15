import numpy as np
import pandas as pd
from bofire.data_models.domain.api import Domain
from bofire.surrogates.trainable import Surrogate, TrainableSurrogate

from bark.bofire_utils.data_models.surrogates import BARKSurrogate as DataModel
from bark.fitting.bark_sampler import run_bark_sampler
from bark.forest import create_empty_forest


class BARKSurrogate(Surrogate, TrainableSurrogate):
    def __init__(self, data_model: DataModel, **kwargs):
        self.warmup_steps = data_model.warmup_steps
        self.num_samples = data_model.num_samples
        self.steps_per_sample = data_model.steps_per_sample
        self.alpha = data_model.alpha
        self.beta = data_model.beta
        self.num_trees = data_model.num_trees
        self.proposal_weights = self._get_proposal_weights(data_model)
        self.verbose = data_model.verbose
        self._init_bark()

        super().__init__(data_model)

    def _init_bark(self):
        self.forest = create_empty_forest(self.num_trees)
        self.noise = np.array([0.1])
        self.scale = np.array([1.0])

    def _get_proposal_weights(self, data_model: DataModel):
        p = np.array(
            [
                data_model.grow_prune_weight,
                data_model.grow_prune_weight,
                data_model.change_weight,
            ]
        )
        return p / np.sum(p)

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        # TODO: use inputs directly
        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        samples = run_bark_sampler(
            (self.forest, self.noise, self.scale),
            (transformed_X.to_numpy(), Y.to_numpy()),
            domain,
            self,
        )
        self.forest, self.noise, self.scale = samples

    def _predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
