import numpy as np

from bark.fitting.tree_proposals import NODE_PROPOSAL_DTYPE
from bark.fitting.tree_traversal import get_node_subspace
from bark.forest import FeatureTypeEnum, create_empty_forest

forest = create_empty_forest(m=50)

bounds = [
    [0.0, 1.0],
    [
        0.0,
        1.0,
    ],
]
feat_types = np.array([FeatureTypeEnum.Cat.value, FeatureTypeEnum.Cat.value])

node_proposal = np.zeros((1,), dtype=NODE_PROPOSAL_DTYPE)[0]
node_proposal["node_idx"] = 0
node_proposal["new_feature_idx"] = 0
node_proposal["new_threshold"] = 0.5
subspace = get_node_subspace(forest[0], 1, bounds, feat_types)
print(subspace)
