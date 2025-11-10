import dataclasses
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from bofire.surrogates.trainable import Surrogate, TrainableSurrogate
from jaxtyping import Float

from bark import forest, types
from bark.fitting.bark_prior_sampler import sample_forest, sample_noise_prior
from bark.fitting.bark_sampler import (
    run_bark_sampler,
)
from bark.tree_kernels.tree_gps import forest_predict, mixture_of_gaussians_as_normal
from bark.utils.bofire import create_data_from_bofire_inputs
from bofire_mixed.data_models.surrogates.bark import (
    BARKPriorSurrogate as BARKPriorSurrogateDataModel,
)
from bofire_mixed.data_models.surrogates.bark import (
    BARKSurrogate as BARKSurrogateDataModel,
)
from bofire_mixed.standardize import Standardize


def _bark_params_to_jax_struct(data_model: BARKSurrogateDataModel):
    keys = [f.name for f in dataclasses.fields(types.BARKConfig)]
    kwargs = {k: v for k, v in data_model.model_dump().items() if k in keys}
    return types.BARKConfig(**kwargs)


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

        self.bark_model: types.BARKModel | None = None
        self.train_data: types.Data | None = None
        self.scaler = Standardize()

        super().__init__(data_model)

    @property
    def is_fitted(self) -> bool:
        """Return True if model is fitted, else False."""
        return self.bark_model is not None

    def _predict(
        self, transformed_X: pd.DataFrame, batched=False, predict_observed=True
    ) -> tuple[Float[np.ndarray, ""], Float[np.ndarray, ""]]:
        candidates = jnp.asarray(transformed_X.to_numpy())
        assert self.bark_model is not None and self.train_data is not None
        mu, var = forest_predict(
            self.bark_model,
            self.train_data,
            candidates,
            diag=True,
        )
        mu, var = self.scaler.untransform_mu_var(mu, var)
        # mu, var are (num_samples, N)
        if predict_observed:
            # y ~ N(f, noise)
            # all observations have the same noise
            var += self.bark_model.noise_var.reshape(-1, 1)

        if not batched:
            mu, var = mixture_of_gaussians_as_normal(mu, var)

        # reshape to ([batch,] n, 1) for the single output
        return np.asarray(mu[..., None]), np.asarray(jnp.sqrt(var[..., None]))

    def _dumps(self):  # type: ignore
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
        self.bark_params = _bark_params_to_jax_struct(data_model)
        self.key = jax.random.key(0)

        super().__init__(data_model)

    def _init_bark(self):
        trees = forest.create_empty_forest(self.num_trees, max_depth=6)
        noise = jnp.array(0.1)
        bark_model = types.BARKModel(trees, noise)

        batch_shape = (self.num_chains, 1)  # num_chains x num_samples_per_chain
        self.bark_model = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None], (*batch_shape, *[1 for _ in x.shape])),
            bark_model,
        )

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = jnp.asarray(
            self.inputs.transform(X, self.input_preprocessing_specs).to_numpy()
        )
        transformed_Y = jnp.asarray(Y.to_numpy())
        transformed_Y = self.scaler(transformed_Y, train=True)

        self.train_data = create_data_from_bofire_inputs(
            transformed_X, transformed_Y, self.inputs
        )

        if not self.is_fitted:
            self._init_bark()
        else:
            # BARK should already be warmed-up from previous iterations
            self.bark_params = replace(self.bark_params, warmup_steps=0)
        # set BARK initialisation from most recent sample
        most_recent_sample = jax.tree_util.tree_map(
            lambda x: x[:, -1, ...], self.bark_model
        )

        self.key, subkey = jax.random.split(self.key)
        self.bark_model = run_bark_sampler(
            most_recent_sample, self.train_data, self.bark_params, subkey
        )


class BARKPriorSurrogate(_BARKSurrogateBase):
    """Samples from the BARK prior distribution."""

    def __init__(self, data_model: BARKPriorSurrogateDataModel, **kwargs):
        self.num_samples = data_model.num_samples
        super().__init__(data_model)
        self.key = jax.random.key(data_model.sample_seed)

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        # we only use a fit method here to store train_data, and to
        # use the same interface as BARKSurrogate
        transformed_X = jnp.asarray(
            self.inputs.transform(X, self.input_preprocessing_specs).to_numpy()
        )
        transformed_Y = jnp.asarray(Y.to_numpy())
        transformed_Y = self.scaler(transformed_Y, train=True)
        self.train_data = create_data_from_bofire_inputs(
            transformed_X, transformed_Y, self.inputs
        )

        params = types.BARKConfig(
            alpha=self.alpha,
            beta=self.beta,
            gamma_prior_shape=self.gamma_prior_shape,
            gamma_prior_rate=self.gamma_prior_rate,
        )

        self.key, forest_key, noise_key = jax.random.split(self.key, num=3)
        forest_key = jax.random.split(forest_key, self.num_samples)
        noise_key = jax.random.split(noise_key, self.num_samples)

        trees = jax.vmap(sample_forest, in_axes=(None, None, None, None, 0))(
            m=self.num_trees,
            bounds=self.train_data.bounds,
            feat_types=self.train_data.feat_types,
            params=params,
            key=forest_key,
        )
        noise = jax.vmap(sample_noise_prior, in_axes=(None, 0))(
            params=params,
            key=noise_key,
        )

        self.bark_model = types.BARKModel(trees=trees, noise_var=noise)
