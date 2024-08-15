import pandas as pd
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import CategoricalInput, NumericalInput

from alfalfa.optimizer.optimizer_utils import get_opt_core_from_domain


def sample_projected(domain: Domain, n: int, seed: int | None = None) -> pd.DataFrame:
    """Sample points from the full input space, then project them to satisfy constraints.

    This does not uniformly sample the feasible space, but is compatible with arbitrary constraints."""
    samples = domain.inputs.sample(n, seed=seed)
    if len(domain.constraints) == 0:
        return samples

    model_core = get_opt_core_from_domain(domain)
    projected_values = []
    model_core.Params.LogToConsole = 0
    model_core.Params.TimeLimit = 5
    model_core.Params.NonConvex = 2

    # TODO: handle sampling categorical variables that are involved in the constraint
    if len(domain.inputs.get(includes=CategoricalInput)):
        raise TypeError(
            "Projected sampling not currently supported for categorical features"
        )

    for _, s in samples.iterrows():
        expr = [
            (s[feat.key] - model_core.getVarByName(feat.key)) ** 2
            for feat in domain.inputs.get(includes=NumericalInput)
        ]

        model_core.setObjective(expr=sum(expr))

        model_core.optimize()

        x_sol = [
            model_core.getVarByName(feat.key).x
            for feat in domain.inputs.get(includes=NumericalInput)
        ]
        projected_values.append(x_sol)

    return pd.DataFrame(data=projected_values, columns=domain.inputs.get_keys())
