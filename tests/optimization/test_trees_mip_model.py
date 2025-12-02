import jax.numpy as jnp

from bark.enums import FeatureTypeEnum
from bark.optimizer.mip_model import TreesMIPModel
from bark.testing.build_test_trees import build_trees_from_dict


def test_build_trees_mip_model():
    tree_dicts = [
        {(0, 0.5): ({(1, 0.75): (None, None)}, None)},
        {(2, 0.2): (None, {(1, 0.75): (None, None)})},
    ]
    tree_dicts = [{(0, 0.5): (None, None)}]
    trees = build_trees_from_dict(tree_dicts, max_depth=3)
    feature_types = jnp.full((len(tree_dicts),), FeatureTypeEnum.Cont)

    trees_mip = TreesMIPModel(trees=trees, feature_types=feature_types)
    print(trees_mip.trees)


test_build_trees_mip_model()
