import numpy as np

# os.environ["NUMBA_DISABLE_JIT"] = "1"
from bark.fitting.tree_proposals import NODE_PROPOSAL_DTYPE, grow
from bark.forest import FeatureTypeEnum, create_empty_forest, forest_gram_matrix

forest = create_empty_forest(m=2)

bounds = [
    [0, 1],
    [0, 1, 2, 3],
]
feat_types = np.array([FeatureTypeEnum.Cat.value, FeatureTypeEnum.Cat.value])

node_proposal = np.zeros((1,), dtype=NODE_PROPOSAL_DTYPE)[0]
node_proposal["node_idx"] = 0
node_proposal["new_feature_idx"] = 1
node_proposal["new_threshold"] = (1 << 1) + (1 << 2)
forest[0] = grow(forest[0], node_proposal)

# subspace = get_node_subspace(forest[0], 1, bounds, feat_types)
# print(subspace)
X = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [1, 0],
    ]
)

K = forest_gram_matrix(forest, X, X, feat_types)
print(K)
