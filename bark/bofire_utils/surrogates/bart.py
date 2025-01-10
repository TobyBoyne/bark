import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from bofire.data_models.features.api import CategoricalInput
from bofire.surrogates.api import Surrogate, TrainableSurrogate

from bark.bofire_utils.data_models.surrogates.bart import (
    BARTSurrogate as BARTSurrogateDataModel,
)
from bark.bofire_utils.standardize import Standardize


class BARTSurrogate(Surrogate, TrainableSurrogate):
    def __init__(self, data_model: BARTSurrogateDataModel):
        self.seed = data_model.seed
        self.scalar = Standardize()
        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None
        super().__init__(data_model)

    @property
    def is_fitted(self) -> bool:
        """Return True if model is fitted, else False."""
        return self.model is not None

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        split_rules = [
            pmb.SubsetSplitRule()
            if isinstance(feat, CategoricalInput)
            else pmb.ContinuousSplitRule()
            for feat in self.inputs.get()
        ]

        # sigma ~ InverseChiSquared(nu=3, t=0.2)
        # t is chosen such that P(sigma < 1) = 0.9
        nu = 3
        t = 0.2

        train_y = self.scalar(Y.to_numpy(), train=True)
        with pm.Model() as bart_model:
            X = pm.Data("X", transformed_X.to_numpy(), mutable=True)
            m = pmb.BART(
                "m", X=X, Y=train_y, m=50, split_rules=split_rules, shape=(1, 100)
            )
            s = pm.InverseGamma("s", alpha=nu / 2, beta=nu * t / 2)
            pm.Normal("y_pred", mu=m, sigma=s, observed=train_y, shape=m.shape)
            train_idata = pm.sample(draws=500, tune=500, random_seed=self.seed)

        self.model = bart_model
        self.trace = train_idata

    def function_samples(self, X: pd.DataFrame) -> np.ndarray:
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        with self.model as bart_model:
            bart_model.set_data("X", transformed_X.to_numpy())
            f_test_draws = pm.sample_posterior_predictive(
                trace=self.trace,
                random_seed=self.seed,
                var_names=["m", "y_pred"],
            )

        return f_test_draws

    def _predict(self, transformed_X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        with self.model as bart_model:
            bart_model.set_data("X", transformed_X.to_numpy())
            f_test_draws = pm.sample_posterior_predictive(
                trace=self.trace,
                random_seed=self.seed,
                var_names=["m", "y_pred"],
            )

        mu = f_test_draws.posterior_predictive.m.mean(dim=["chain"])
        var = f_test_draws.posterior_predictive.y_pred.var(dim=["chain"])
        mu, var = self.scalar.untransform_mu_var(mu, var)
        return mu, np.sqrt(var)
        # f_test = f_test_draws.posterior_predictive.m.mean(dim=["chain"])
        # acq = np.maximum(train_y.min() - f_test, 0.0).mean(dim=["draw"])

    def _dumps(self):
        pass

    def loads(self, data: str):
        pass
