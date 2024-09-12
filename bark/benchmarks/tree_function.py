import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bofire.benchmarks.api import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import MinimizeObjective

from bark.fitting.tree_proposals import NODE_PROPOSAL_DTYPE, grow
from bark.forest import FeatureTypeEnum, create_empty_forest, pass_through_forest


def sample_tree_function_from_structure(forest):
    """Sample a tree function $f(x)=\sum_j g(x; T_j)$"""

    rng_leaf_value = np.random.default_rng(seed=0)
    leaf_values = rng_leaf_value.standard_normal(forest.shape)

    def f(x):
        feat_types = np.full(x.shape[1], FeatureTypeEnum.Cont.value)
        leaves = pass_through_forest(forest, x, feat_types)
        g_leaf_values = leaf_values[np.arange(leaf_values.shape[0]), leaves]
        f_value = g_leaf_values.sum(axis=1)
        return f_value

    return f


def sample_tree_structure_from_prior(m: int, domain: Domain):
    forest = create_empty_forest(m)
    alpha = 0.95
    beta = 2.0
    dim = len(domain.inputs)
    rng = np.random.default_rng(seed=0)
    for tree in forest:
        new_nodes = [0]
        while new_nodes:
            node_idx = new_nodes.pop()
            depth = tree[node_idx]["depth"]
            depth_prior = alpha * (1 + depth) ** (-beta)
            if rng.uniform() > depth_prior:
                continue
            node_proposal = np.zeros((1,), dtype=NODE_PROPOSAL_DTYPE)[0]
            node_proposal["node_idx"] = node_idx
            node_proposal["new_feature_idx"] = rng.integers(dim)
            node_proposal["new_threshold"] = rng.uniform(0, 1)

            grow(tree, node_proposal)
            new_nodes.extend([tree[node_idx]["left"], tree[node_idx]["right"]])

    return forest


class TreeFunction(Benchmark):
    """A function sample from a BARK prior.

    This is a good test that BARK is indeed able to optimize on tree functions."""

    def __init__(self, dim=5, m=50, **kwargs):
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    *(
                        ContinuousInput(key=f"x_{i}", bounds=(0.0, 1.0))
                        for i in range(dim)
                    )
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )

        forest = sample_tree_structure_from_prior(m, self._domain)
        self._tree_func = sample_tree_function_from_structure(forest)

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ys = self._tree_func(X.to_numpy())
        return pd.DataFrame(data=ys, columns=self.domain.outputs.get_keys())


if __name__ == "__main__":
    benchmark = TreeFunction(dim=1, m=10)
    x = pd.DataFrame(
        data=np.linspace(0, 1, 100)[:, None], columns=benchmark.domain.inputs.get_keys()
    )
    y = benchmark.f(x)
    plt.step(x, y)
    x = pd.DataFrame(
        data=np.linspace(0.01, 1.02, 100)[:, None],
        columns=benchmark.domain.inputs.get_keys(),
    )
    y = benchmark.f(x)
    plt.step(x, y)
    plt.show()
