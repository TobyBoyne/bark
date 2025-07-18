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

from bark.fitting.tree_proposals import NodeProposal, grow
from bark.forest import create_empty_forest, pass_through_forest
from bofire_mixed.domain import get_feature_types_array


def sample_tree_function_from_structure(
    forest: np.ndarray, domain: Domain, tree_rng: np.random.Generator
):
    """Sample a tree function $f(x)=\sum_j g(x; T_j)$"""

    leaf_values = tree_rng.standard_normal(forest.shape)
    feat_types = get_feature_types_array(domain)

    def f(x):
        leaves = pass_through_forest(forest, x, feat_types)
        g_leaf_values = leaf_values[np.arange(leaf_values.shape[0]), leaves]
        f_value = g_leaf_values.sum(axis=1)
        return f_value

    return f


def sample_tree_structure_from_prior(m: int, domain: Domain, rng: np.random.Generator):
    forest = create_empty_forest(m)
    alpha = 0.95
    beta = 2.0
    dim = len(domain.inputs)
    for tree in forest:
        new_nodes = [0]
        while new_nodes:
            node_idx = new_nodes.pop()
            depth = tree[node_idx]["depth"]
            depth_prior = alpha * (1 + depth) ** (-beta)
            if rng.uniform() > depth_prior:
                continue
            node_proposal = NodeProposal()
            node_proposal.node_idx = node_idx
            node_proposal.new_feature_idx = rng.integers(dim)
            node_proposal.new_threshold = rng.uniform(0, 1)

            grow(tree, node_proposal)
            new_nodes.extend([tree[node_idx]["left"], tree[node_idx]["right"]])

    return forest


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
        forest = sample_tree_structure_from_prior(m, self._domain, rng)
        self._tree_func = sample_tree_function_from_structure(forest, self._domain, rng)

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
