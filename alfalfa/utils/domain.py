"""Utils for working with Bofire domains"""
import gurobipy
from bofire.data_models.constraints.api import (
    AnyConstraint,
    EqalityConstraint,
    LinearConstraint,
)
from bofire.data_models.domain.api import Domain, Features
from bofire.data_models.features.api import (
    AnyFeature,
    CategoricalInput,
    ContinuousInput,
    DiscreteInput,
)


def get_feature_by_index(features: Features, index: int) -> AnyFeature:
    return features.get().features[index]


def get_index_by_feature_key(features: Features, key: str) -> int:
    return features.get().features.index(features.get_by_key(key))


def get_feature_bounds(feature: AnyFeature) -> tuple[float, float] | list[str]:
    if isinstance(feature, CategoricalInput):
        return feature.categories
    elif isinstance(feature, DiscreteInput):
        return (feature.lower_bound, feature.upper_bound)
    elif isinstance(feature, ContinuousInput):
        return feature.bounds

    raise TypeError(f"Cannot get bounds for feature of type {feature.type}")


def get_cat_idx_from_domain(domain: Domain) -> set[int]:
    """Get the indices of categorical features"""
    return {
        i
        for i, feat in enumerate(domain.inputs.get())
        if isinstance(feat, CategoricalInput)
    }


def build_integer_input(*, key: str, unit: str | None = None, bounds: tuple[int, int]):
    lb, ub = bounds
    values = list(range(lb, ub + 1))
    return DiscreteInput(key=key, unit=unit, values=values)


def apply_constraint_to_model(
    constraint: AnyConstraint, model_core: gurobipy.Model
) -> gurobipy.GenExpr:
    if isinstance(constraint, (LinearConstraint)):
        expr = gurobipy.quicksum(
            (
                model_core.getVarByName(key) * coeff
                for key, coeff in zip(constraint.features, constraint.coefficients)
            )
        )
        rhs = constraint.rhs
        gur_constr = (
            (expr == rhs)
            if isinstance(constraint, EqalityConstraint)
            else (expr <= rhs)
        )
        return model_core.addConstr(gur_constr)
