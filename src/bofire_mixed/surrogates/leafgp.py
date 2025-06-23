import bofire.priors.api as priors
import botorch
import numpy as np
import pandas as pd
import torch
from bofire.data_models.domain.api import Domain
from bofire.surrogates.api import Surrogate, TrainableSurrogate
from bofire.utils.torch_tools import tkwargs
from botorch import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bark.bofire_utils.data_models.surrogates.leafgp import (
    LeafGPSurrogate as LeafGPDataModel,
)
from bark.fitting.lgbm_fitting import fit_lgbm_forest, lgbm_to_bark_forest
from bark.tree_kernels.tree_model_kernel import TreeAgreementKernel
from bofire_mixed.domain import get_feature_types_array

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
        model = (
            self.model.covar_module.base_kernel.forest,
            self.model.likelihood.noise.item(),
            self.model.covar_module.outputscale.item(),
        )
        return None if any(x is None for x in model) else model

    @property
    def is_fitted(self) -> bool:
        """Return True if model is fitted, else False."""
        return self.model_as_tuple() is not None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        booster = fit_lgbm_forest(transformed_X, Y, domain)  # TODO: add params
        assert booster.num_trees() > 1, "Insufficient data to train forest kernel."
        forest = lgbm_to_bark_forest(booster)

        self.model = botorch.models.SingleTaskGP(
            train_X=tX,
            train_Y=tY,
            covar_module=ScaleKernel(
                TreeAgreementKernel(
                    forest=forest,
                    feat_types=get_feature_types_array(domain),
                )
            ),
            outcome_transform=(Standardize(m=tY.shape[-1])),
            input_transform=Normalize(d=transformed_X.shape[-1]),
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)

    def _predict(self, transformed_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = torch.from_numpy(transformed_X.to_numpy()).to(**tkwargs)
        with torch.no_grad():
            preds = (
                self.model.posterior(X=X, observation_noise=True)
                .mean.cpu()
                .detach()
                .numpy()
            )
            stds = np.sqrt(
                self.model.posterior(X=X, observation_noise=True)
                .variance.cpu()
                .detach()
                .numpy(),
            )
        return preds, stds

    @property
    def train_data(self):
        return (
            self.model.train_inputs[0].detach().cpu().numpy(),
            self.model.train_targets.detach().cpu().numpy().reshape(-1, 1),
        )

    def _dumps(self):
        pass

    def loads(self, data: str):
        pass
