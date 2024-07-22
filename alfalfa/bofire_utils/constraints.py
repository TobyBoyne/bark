from typing import Annotated, Callable, List, Literal, Tuple

import gurobipy
import numpy as np
import pandas as pd
from bofire.data_models.constraints.api import (
    AnyConstraint,
    EqalityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
    LinearConstraint,
    NonlinearConstraint,
)
from bofire.data_models.types import make_unique_validator
from pydantic import AfterValidator, Field, model_validator

QuadraticFeatureKeys = Annotated[
    List[Tuple[str, str]],
    Field(min_length=2),
    AfterValidator(make_unique_validator("QuadraticFeatures")),
]

ConstraintFuncT = Callable[[list[float | gurobipy.Var], gurobipy.Model | None], float]

GurobiExpressionT = gurobipy.LinExpr | gurobipy.QuadExpr | gurobipy.GenExpr
GurobiConstraintT = gurobipy.Constr | gurobipy.QConstr


class QuadraticConstraint(IntrapointConstraint):
    """Abstract base class for quadratic equality and inequality constraints.

    Attributes:
        features (list): list of feature keys (str) on which the constraint works on.
        coefficients (list): list of coefficients (float) of the constraint.
        rhs (float): Right-hand side of the constraint
    """

    type: Literal["QuadraticConstraint"] = "QuadraticConstraint"

    features: QuadraticFeatureKeys
    coefficients: Annotated[List[float], Field(min_length=2)]
    rhs: float

    @model_validator(mode="after")
    def validate_list_lengths(self):
        """Validate that length of the feature and coefficient lists have the same length."""
        if len(self.features) != len(self.coefficients):
            raise ValueError(
                f"must provide same number of features and coefficients, got {len(self.features)} != {len(self.coefficients)}"
            )
        return self

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        a, b = map(list, zip(*self.features))
        data = experiments[a].to_numpy() * experiments[b].to_numpy()
        products = pd.DataFrame(data=data, index=experiments.index)
        return (products @ self.coefficients - self.rhs) / np.linalg.norm(
            np.array(self.coefficients)
        )

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class FunctionalConstraint(NonlinearConstraint):
    """Arbitrary constraint that takes any functional form."""

    type: Literal["FunctionalConstraint"] = "FunctionalConstraint"
    func: ConstraintFuncT
    rhs: float
    expression: str = ""

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        exp_as_idx_cols = experiments.rename(
            columns={col: i for i, col in enumerate(experiments.columns)}
        )
        return exp_as_idx_cols.apply(self.func, axis="columns") - self.rhs

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class QuadraticEqualityConstraint(QuadraticConstraint, EqalityConstraint):
    type: Literal["QuadraticEqualityConstraint"] = "QuadraticEqualityConstraint"


class QuadraticInequalityConstraint(QuadraticConstraint, InequalityConstraint):
    type: Literal["QuadraticInequalityConstraint"] = "QuadraticInequalityConstraint"


class FunctionalEqualityConstraint(FunctionalConstraint, EqalityConstraint):
    type: Literal["FunctionalEqualityConstraint"] = "FunctionalEqualityConstraint"


class FunctionalInequalityConstraint(FunctionalConstraint, InequalityConstraint):
    type: Literal["FunctionalInequalityConstraint"] = "FunctionalInequalityConstraint"


ExtendedAnyConstraint = (
    AnyConstraint
    | QuadraticEqualityConstraint
    | QuadraticInequalityConstraint
    | FunctionalEqualityConstraint
    | FunctionalInequalityConstraint
)


def _expr_to_equality(expr: GurobiExpressionT, constraint: ExtendedAnyConstraint):
    rhs = constraint.rhs
    return (expr == rhs) if isinstance(constraint, EqalityConstraint) else (expr <= rhs)


def apply_constraint_to_model(
    constraint: ExtendedAnyConstraint, model_core: gurobipy.Model
) -> GurobiConstraintT:
    if isinstance(constraint, LinearConstraint):
        expr = gurobipy.quicksum(
            (
                model_core.getVarByName(key) * coeff
                for key, coeff in zip(constraint.features, constraint.coefficients)
            )
        )
        gur_constr = _expr_to_equality(expr, constraint)
        return model_core.addConstr(gur_constr)

    elif isinstance(constraint, QuadraticConstraint):
        expr = gurobipy.quicksum(
            (
                model_core.getVarByName(key1) * model_core.getVarByName(key2) * coeff
                for (key1, key2), coeff in zip(
                    constraint.features, constraint.coefficients
                )
            )
        )
        gur_constr = _expr_to_equality(expr, constraint)
        return model_core.addConstr(gur_constr)

    elif isinstance(constraint, FunctionalConstraint):
        expr = constraint.func(model_core._cont_var_dict, model_core)
        gur_constr = _expr_to_equality(expr, constraint)
        return model_core.addConstr(gur_constr)

    raise NotImplementedError(f"{constraint.type} is not yet supported.")
