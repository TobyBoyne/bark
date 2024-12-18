import botorch
import numpy as np
import pandas as pd
import torch
from bofire.data_models.domain.api import Domain
from bofire.surrogates.trainable import Surrogate, TrainableSurrogate
from bofire.utils.torch_tools import tkwargs

from bark.bofire_utils.data_models.surrogates import (
    BARKSurrogate as BARKSurrogateDataModel,
)
from bark.bofire_utils.data_models.surrogates import LeafGPSurrogate as LeafGPDataModel
from bark.bofire_utils.domain import get_feature_types_array
from bark.fitting.bark_sampler import (
    BARK_JITCLASS_SPEC,
    BARKTrainParamsNumba,
    run_bark_sampler,
)
from bark.fitting.lgbm_fitting import fit_lgbm_forest, lgbm_to_bark_forest
from bark.forest import create_empty_forest
from bark.tree_kernels.tree_gps import forest_predict, mixture_of_gaussians_as_normal
from bark.tree_kernels.tree_model_kernel import BARKTreeModelKernel


def _bark_params_to_jitclass(data_model: BARKSurrogateDataModel):
    proposal_weights = np.array(
        [
            data_model.grow_prune_weight,
            data_model.grow_prune_weight,
            data_model.change_weight,
        ]
    )
    proposal_weights /= np.sum(proposal_weights)

    keys = list(zip(*BARK_JITCLASS_SPEC))[0]
    kwargs = {k: v for k, v in data_model.model_dump().items() if k in keys}
    return BARKTrainParamsNumba(proposal_weights=proposal_weights, **kwargs)


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


class BARKSurrogate(Surrogate, TrainableSurrogate):
    def __init__(self, data_model: BARKSurrogateDataModel, **kwargs):
        self.warmup_steps = data_model.warmup_steps
        self.num_samples = data_model.num_samples
        self.steps_per_sample = data_model.steps_per_sample
        self.alpha = data_model.alpha
        self.beta = data_model.beta
        self.num_chains = data_model.num_chains
        self.num_trees = data_model.num_trees
        self.verbose = data_model.verbose
        self.use_softplus_transform = data_model.use_softplus_transform
        self.sample_scale = data_model.sample_scale
        self.gamma_prior_shape = data_model.gamma_prior_shape
        self.gamma_prior_rate = data_model.gamma_prior_rate
        self.bark_params = _bark_params_to_jitclass(data_model)

        self.forest = None
        self.noise = None
        self.scale = None
        self.train_data = None

        super().__init__(data_model)

    def _init_bark(self):
        forest = create_empty_forest(self.num_trees)

        self.forest = np.tile(forest, (self.num_chains, 1, 1, 1))
        self.noise = np.tile(0.1, (self.num_chains, 1))
        self.scale = np.tile(1.0, (self.num_chains, 1))

    def model_as_tuple(self) -> None | tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = (self.forest, self.noise, self.scale)
        return None if any(x is None for x in model) else model

    @property
    def is_fitted(self) -> bool:
        """Return True if model is fitted, else False."""
        return self.model_as_tuple() is not None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        # TODO: use inputs directly
        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        Y = Y.to_numpy()
        Y_standardized = (Y - Y.mean()) / Y.std()
        self.train_data = (transformed_X.to_numpy(), Y_standardized)

        if not self.is_fitted:
            self._init_bark()
        else:
            # BARK should already be warmed-up from previous iterations
            self.bark_params.warmup_steps = 0
        # set BARK initialisation from most recent sample
        most_recent_sample = (
            self.forest[:, -1, :, :],
            self.noise[:, -1],
            self.scale[:, -1],
        )

        samples = run_bark_sampler(
            most_recent_sample,
            self.train_data,
            domain,
            self.bark_params,
        )
        self.forest, self.noise, self.scale = samples

    def _predict(self, transformed_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        candidates = transformed_X.to_numpy()
        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        mu, var = forest_predict(
            self.model_as_tuple(),
            self.train_data,
            candidates,
            domain,
            diag=True,
        )
        mu_f, var_f = mixture_of_gaussians_as_normal(mu, var)
        # reshape to (n, 1) for the single output
        return mu_f.reshape(-1, 1), var_f.reshape(-1, 1)

    def _dumps(self):
        pass

    def loads(self, data: str):
        pass


class BARKPriorSurrogate(Surrogate, TrainableSurrogate):
    """Samples from the BARK prior distribution."""
