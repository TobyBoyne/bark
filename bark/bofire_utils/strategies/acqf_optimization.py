"""Mixed acquisition function optimization"""

from typing import List, Optional, Tuple

import pandas as pd
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, DiscreteInput
from bofire.data_models.types import InputTransformSpecs
from bofire.strategies.predictives.acqf_optimization import BotorchOptimizer
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_mixed_alternating
from torch import Tensor


class AlternatingBotorchOptimizer(BotorchOptimizer):
    def _optimize(
        self,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        domain: Domain,
        input_preprocessing_specs: InputTransformSpecs,
        experiments: Optional[pd.DataFrame] = None,
    ) -> Tuple[Tensor, Tensor]:
        bounds = self.get_bounds(domain, input_preprocessing_specs)
        discrete_keys = domain.inputs.get_keys(includes=DiscreteInput)
        categorical_keys = domain.inputs.get_keys(includes=CategoricalInput)
        discrete_dims = domain.inputs.get_feature_indices(
            input_preprocessing_specs, discrete_keys
        )
        cat_dims = domain.inputs.get_feature_indices(
            input_preprocessing_specs, categorical_keys
        )

        candidates, _ = optimize_acqf_mixed_alternating(
            acq_function=acqfs[0],
            bounds=bounds,
            q=candidate_count,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            discrete_dims=discrete_dims,
            cat_dims=cat_dims,
        )

        candidates = self._candidates_tensor_to_dataframe(
            candidates, domain, input_preprocessing_specs
        )

        return candidates
