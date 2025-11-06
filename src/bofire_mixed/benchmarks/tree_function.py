import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective

# from bark.fitting.bark_prior_sampler import sample_forest_prior
from bark.forest import create_empty_forest, pass_through_forest
from bofire_mixed.domain import get_feature_bounds, get_feature_types_array


def sample_tree_function_from_structure(
    nodes: np.ndarray, domain: Domain, tree_rng: np.random.Generator
):
    """Sample a tree function $f(x)=\sum_j g(x; T_j)$"""

    leaf_values = tree_rng.standard_normal(nodes.shape)
    feat_types = get_feature_types_array(domain)

    def f(x):
        leaves = pass_through_forest(nodes, x, feat_types)
        g_leaf_values = leaf_values[np.arange(leaf_values.shape[0]), leaves]
        f_value = g_leaf_values.sum(axis=1)
        return f_value

    return f


def sample_tree_structure_from_prior(m: int, domain: Domain, rng: np.random.Generator):
    nodes = create_empty_forest(m)
    alpha = 0.95
    beta = 2.0
    bounds = np.array(
        [get_feature_bounds(feat, encoding="bitmask") for feat in domain.inputs.get()]
    )
    feat_types = get_feature_types_array(domain)
    nodes = sample_forest_prior(
        m, bounds, feat_types, alpha, beta, num_samples=1, rng=rng
    )

    return nodes


class TreeFunction(Benchmark):
    """A function sample from a BARK prior.

    This is a good test that BARK is indeed able to optimize on tree functions."""

    def __init__(self, dim=5, cat_dim=0, num_cat=5, m=50, function_seed=1, **kwargs):
        super().__init__(**kwargs)
        categories = [chr(i + ord("a")) for i in range(num_cat)]
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        ContinuousInput(key=f"x_{i}", bounds=(0.0, 1.0))
                        for i in range(dim)
                    ),
                    *(
                        CategoricalInput(key=f"c_{i}", categories=categories)
                        for i in range(cat_dim)
                    ),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

        rng = np.random.default_rng(function_seed)
        nodes = sample_tree_structure_from_prior(m, self._domain, rng)
        self._tree_func = sample_tree_function_from_structure(nodes, self._domain, rng)

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        specs = {
            k: CategoricalEncodingEnum.ORDINAL
            for k in self.domain.inputs.get_keys(includes=CategoricalInput)
        }
        X_transformed = self.domain.inputs.transform(X, specs).to_numpy()
        ys = self._tree_func(X_transformed)
        return pd.DataFrame(data=ys, columns=self.domain.outputs.get_keys())


if __name__ == "__main__":
    benchmark = TreeFunction(dim=1, m=10)
    x = pd.DataFrame(
        data=np.linspace(0, 1, 1000)[:, None],
        columns=benchmark.domain.inputs.get_keys(),
    )
    y = benchmark.f(x)
    plt.step(x, y)
    x = pd.DataFrame(
        data=np.linspace(0.01, 1.02, 1000)[:, None],
        columns=benchmark.domain.inputs.get_keys(),
    )
    y = benchmark.f(x)
    plt.step(x, y)
    plt.show()
