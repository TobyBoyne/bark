import numpy as np
import pandas as pd
from bofire.data_models.domain.api import Domain
from bofire.surrogates.trainable import Surrogate, TrainableSurrogate

from bark.fitting.bark_prior_sampler import sample_forest_prior, sample_noise_prior
from bark.fitting.bark_sampler import (
    BARK_JITCLASS_SPEC,
    BARKTrainParamsNumba,
    run_bark_sampler,
)
from bark.forest import create_empty_forest
from bark.tree_kernels.tree_gps import forest_predict, mixture_of_gaussians_as_normal
from bofire_mixed.data_models.surrogates.bark import (
    BARKPriorSurrogate as BARKPriorSurrogateDataModel,
)
from bofire_mixed.data_models.surrogates.bark import (
    BARKSurrogate as BARKSurrogateDataModel,
)
from bofire_mixed.domain import get_feature_bounds, get_feature_types_array
from bofire_mixed.standardize import Standardize


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


class _BARKSurrogateBase(Surrogate, TrainableSurrogate):
    def __init__(
        self, data_model: BARKSurrogateDataModel | BARKPriorSurrogateDataModel
    ):
        self.alpha = data_model.alpha
        self.beta = data_model.beta
        self.num_trees = data_model.num_trees

        self.alpha = data_model.alpha
        self.beta = data_model.beta
        self.num_trees = data_model.num_trees

        self.gamma_prior_shape = data_model.gamma_prior_shape
        self.gamma_prior_rate = data_model.gamma_prior_rate

        self.forest = None
        self.noise = None
        self.scale = None
        self.train_data = None
        self.scaler = Standardize()

        super().__init__(data_model)

    def model_as_tuple(self) -> None | tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = (self.forest, self.noise, self.scale)
        return None if any(x is None for x in model) else model

    @property
    def is_fitted(self) -> bool:
        """Return True if model is fitted, else False."""
        return self.model_as_tuple() is not None

    def _predict(
        self, transformed_X: pd.DataFrame, batched=False, predict_observed=True
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = transformed_X.to_numpy()
        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        mu, var = forest_predict(
            self.model_as_tuple(),
            self.train_data,
            candidates,
            domain,
            diag=True,
        )
        mu, var = self.scaler.untransform_mu_var(mu, var)
        # mu, var are (num_samples, N)
        if predict_observed:
            # y ~ N(f, noise)
            # all observations have the same noise
            var += self.noise.reshape(-1, 1)

        if not batched:
            mu, var = mixture_of_gaussians_as_normal(mu, var)

        # reshape to ([batch,] n, 1) for the single output
        return mu[..., np.newaxis], np.sqrt(var[..., np.newaxis])

    def _dumps(self):
        pass

    def loads(self, data: str):
        pass


class BARKSurrogate(_BARKSurrogateBase):
    def __init__(self, data_model: BARKSurrogateDataModel, **kwargs):
        self.warmup_steps = data_model.warmup_steps
        self.num_samples = data_model.num_samples
        self.steps_per_sample = data_model.steps_per_sample
        self.num_chains = data_model.num_chains
        self.verbose = data_model.verbose
        self.use_softplus_transform = data_model.use_softplus_transform
        self.sample_scale = data_model.sample_scale
        self.bark_params = _bark_params_to_jitclass(data_model)

        super().__init__(data_model)

    def _init_bark(self):
        forest = create_empty_forest(self.num_trees)

        self.forest = np.tile(forest, (self.num_chains, 1, 1, 1))
        self.noise = np.tile(0.1, (self.num_chains, 1))
        self.scale = np.tile(1.0, (self.num_chains, 1))

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        # TODO: use inputs directly
        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        Y = Y.to_numpy()
        Y_standardized = self.scaler(Y, train=True)
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


class BARKPriorSurrogate(_BARKSurrogateBase):
    """Samples from the BARK prior distribution."""

    def __init__(self, data_model: BARKPriorSurrogateDataModel, **kwargs):
        self.num_samples = data_model.num_samples
        super().__init__(data_model)
        self.sample_rng = np.random.default_rng(data_model.sample_seed)

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        # we only use a fit method here to store train_data, and to
        # use the same interface as BARKSurrogate
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        domain = Domain(inputs=self.inputs, outputs=self.outputs)
        Y = Y.to_numpy()
        Y_standardized = self.scaler(Y, train=True)
        self.train_data = (transformed_X.to_numpy(), Y_standardized)

        bounds = np.array(
            [get_feature_bounds(feat, encoding="bitmask") for feat in self.inputs.get()]
        )
        feat_types = get_feature_types_array(domain)

        self.forest = sample_forest_prior(
            m=self.num_trees,
            bounds=bounds,
            feat_types=feat_types,
            alpha=self.alpha,
            beta=self.beta,
            num_samples=self.num_samples,
            rng=self.sample_rng,
        )
        self.noise = sample_noise_prior(
            gamma_shape=self.gamma_prior_shape,
            gamma_rate=self.gamma_prior_rate,
            num_samples=self.num_samples,
            rng=self.sample_rng,
        )
        self.scale = np.ones((self.num_samples,))
