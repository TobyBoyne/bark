"""Mixed acquisition function optimization"""

from typing import Callable, Dict, List, Optional, Tuple

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import DiscreteInput
from bofire.strategies.predictives.acqf_optimization import BotorchOptimizer
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_mixed_alternating
from torch import Tensor


class AlternatingBotorchOptimizer(BotorchOptimizer):
    def _optimize_acqf_continuous(
        self,
        domain: Domain,
        candidate_count: int,
        acqfs: List[AcquisitionFunction],
        bounds: Tensor,
        ic_generator: Callable,
        ic_gen_kwargs: Dict,
        nonlinear_constraints: List[Callable[[Tensor], float]],
        fixed_features: Optional[Dict[int, float]],
        fixed_features_list: Optional[List[Dict[int, float]]],
        sequential: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        # this is not correct, as specs will matter if eg. there is a one-hot encoding
        specs = {}
        discrete_keys = domain.inputs.get_keys(includes=DiscreteInput)
        discrete_dims = domain.inputs.get_feature_indices(specs, discrete_keys)
        return optimize_acqf_mixed_alternating(
            acq_function=acqfs[0],
            bounds=bounds,
            q=candidate_count,
            num_restarts=self.n_restarts,
            raw_samples=self.n_raw_samples,
            discrete_dims=discrete_dims,
        )
