from typing import Literal, Type

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy

from bark.bofire_utils.data_models.surrogates.bark import (
    BARKPriorSurrogate,
    BARKSurrogate,
)
from bark.bofire_utils.data_models.surrogates.leafgp import LeafGPSurrogate

AnyTreeSurrogate = BARKSurrogate | BARKPriorSurrogate | LeafGPSurrogate


class TreeKernelStrategy(PredictiveStrategy):
    """Strategy for tree-kernel models.

    This strategy is used for tree-kernel surrogates, including LeafGP
    and BARK."""

    type: Literal["TreeKernelStrategy"] = "TreeKernelStrategy"

    surrogate_specs: AnyTreeSurrogate  # = Field(
    #    default_factory=lambda: TreeKernelStrategy, validate_default=True
    # )

    @staticmethod
    def _generate_surrogate_specs(
        domain: Domain,
        surrogate_specs: BARKSurrogate,
    ) -> BARKSurrogate:
        """Method to generate model specifications when no model specs are passed

        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            surrogate_specs (List[ModelSpec], optional): List of model specification classes specifying the models to be used in the strategy. Defaults to None.

        Raises:
            KeyError: if there is a model spec for an unknown output feature
            KeyError: if a model spec has an unknown input feature
        Returns:
            List[ModelSpec]: List of model specification classes

        """
        return surrogate_specs

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            ContinuousInput,
            ContinuousOutput,
        ]

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return my_type in [MinimizeObjective, MaximizeObjective]
