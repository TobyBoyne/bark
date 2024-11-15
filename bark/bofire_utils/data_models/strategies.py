from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from pydantic import Field

from bark.bofire_utils.data_models.surrogates import BARKSurrogate


class TreeKernelStrategy(PredictiveStrategy):
    """Strategy for tree-kernel models.

    This strategy is used for tree-kernel surrogates, including LeafGP
    and BARK."""

    surrogate_specs: BARKSurrogate = Field(
        default_factory=lambda: BARKSurrogate(), validate_default=True
    )
