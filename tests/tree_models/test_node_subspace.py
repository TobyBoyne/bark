import numpy as np

from bark.fitting.tree_proposals import NODE_PROPOSAL_DTYPE, grow
from bark.fitting.tree_traversal import get_node_subspace
from bark.forest import FeatureTypeEnum, create_empty_forest

forest = create_empty_forest(m=2)

bounds = np.array(
    [
        (0, 0b11),
        (0, 0b1111),
    ]
)


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

sub = get_node_subspace(forest[0], 1, bounds, feat_types)
print(f"{sub[1, 1]:b}")

cat_threshold = [
    i for i in range(int(sub[1, 1]).bit_length()) if (int(sub[1, 1]) >> i) & 1
]
print(cat_threshold)
