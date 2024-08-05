from collections.abc import Callable
from typing import Annotated, List, Literal, Tuple

import gurobipy
import numpy as np
import pandas as pd
import sympy
from bofire.data_models.constraints.api import (
    AnyConstraint,
    EqalityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
    LinearConstraint,
    NonlinearConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.types import make_unique_validator
from pydantic import AfterValidator, Field, model_validator
from sympy.core.relational import Equality, LessThan

EvaluatedSymbolT = gurobipy.GenExpr | gurobipy.Var | gurobipy.LinExpr | float


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


def _get_var_or_create(
    model_core: gurobipy.Model,
    args: list[EvaluatedSymbolT],
    name_format: str,
    escape_float=False,
    **var_kwargs,
) -> gurobipy.Var:
    """Get or create a variable in the model core based on the arguments.
    Returns None if no auxilliary variable is needed."""
    if escape_float and any(isinstance(arg, (float, sympy.Float)) for arg in args):
        return None
    args_names = [v.VarName if isinstance(v, gurobipy.Var) else str(v) for v in args]
    name = name_format.format(*args_names)
    gur_var = model_core.getVarByName(name_format)
    if gur_var is None:
        gur_var = model_core.addVar(name=name, **var_kwargs)
    return gur_var


def _recurse_sympy(
    expr: sympy.Expr, model_core: gurobipy.Model
) -> gurobipy.GenExpr | gurobipy.Var | gurobipy.LinExpr | float:
    if expr.is_Add:
        return gurobipy.quicksum(_recurse_sympy(arg, model_core) for arg in expr.args)
    elif expr.is_Mul:
        args = [_recurse_sympy(arg, model_core) for arg in expr.args]

        arg_prod = _get_var_or_create(
            model_core, args, name_format="_({}*{})", escape_float=True
        )
        if arg_prod is None:
            return args[0] * args[1]
        model_core.addConstr(arg_prod == args[0] * args[1])
        model_core.update()
        if len(expr.args) == 2:
            return arg_prod
        rest_prod = _recurse_sympy(sympy.Mul(*expr.args[2:]), model_core)
        all_prod = _get_var_or_create(
            model_core, [arg_prod, rest_prod], name_format="_({}*{})"
        )
        model_core.addConstr(all_prod == arg_prod * rest_prod)
        model_core.update()

        return all_prod
    elif expr.is_Number:
        return expr.evalf()
    elif expr.is_Symbol:
        return model_core.getVarByName(expr.name)
    elif expr.is_Pow:
        args = [_recurse_sympy(arg, model_core) for arg in expr.args]
        arg_exp = _get_var_or_create(model_core, args, name_format="_{}^{}")
        model_core.addGenConstrPow(xvar=args[0], yvar=arg_exp, a=args[1])
        model_core.update()
        return arg_exp
    else:
        raise NotImplementedError(f"{expr} is not yet supported.")


def add_constr_expr_to_gurobipy_model(
    expression_string: str, model_core: gurobipy.Model, domain: Domain
):
    features = domain.inputs.get_keys()
    symbols = sympy.symbols(features)
    key_to_symbol = {key: symbol for key, symbol in zip(features, symbols)}

    expr = sympy.parse_expr(expression_string, key_to_symbol)
    assert isinstance(
        expr, (LessThan, Equality)
    ), f"{expr} is not a relational expression."
    lhs, rhs = expr.args
    gur_expr = _recurse_sympy(lhs, model_core)
    gur_constr = gur_expr <= rhs if isinstance(expr.func, LessThan) else gur_expr == rhs
    model_core.addConstr(gur_constr)
