import botorch
import pandas as pd
import torch
from bofire.data_models.domain.api import Domain
from bofire.surrogates.trainable import Surrogate, TrainableSurrogate
from bofire.utils.torch_tools import tkwargs

from bark.bofire_utils.data_models.surrogates.leafgp import (
    LeafGPSurrogate as LeafGPDataModel,
)
from bark.bofire_utils.domain import get_feature_types_array
from bark.fitting.lgbm_fitting import fit_lgbm_forest, lgbm_to_bark_forest

# from bark.bofire_utils.standardize import Standardize


class LeafGPSurrogate(Surrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: LeafGPDataModel,
        **kwargs,
    ):
        self.noise_prior = data_model.noise_prior
        super().__init__(data_model=data_model, **kwargs)

    model: botorch.models.SingleTaskGP | None = None
    training_specs: dict = {}

    def model_as_tuple(self):
        return (
            self.forest,
            self.model.likelihood.noise.item(),
            self.model.covar_module.outputscale.item(),
        )

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        booster = fit_lgbm_forest(transformed_X, Y, self.domain)  # TODO: add params
        forest = lgbm_to_bark_forest(booster)

        self.model = botorch.models.SingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=BARKTreeModelKernel(
                forest=forest,
                feat_types=get_feature_types_array(domain),
            ),
            outcome_transform=(
                Standardize(m=tY.shape[-1])
                if self.output_scaler == ScalerEnum.STANDARDIZE
                else None
            ),
            input_transform=scaler,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
