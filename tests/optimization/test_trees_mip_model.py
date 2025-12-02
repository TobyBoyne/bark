import jax.numpy as jnp
import pytest

from bark.enums import FeatureTypeEnum
from bark.optimizer.mip_model import MIPDecisionNode, MIPLeaf, TreesMIPModel
from bark.testing.build_test_trees import TreeDict, build_trees_from_dict


@pytest.mark.parametrize(
    argnames=("tree_dicts", "target_trees_mip"),
    argvalues=[
        (
            [
                None,
                {(0, 0.5): (None, None)},
                None,
            ],
            [
                MIPDecisionNode(
                    left=MIPLeaf(),
                    right=MIPLeaf(),
                    feature_idx=0,
                    feature_type=FeatureTypeEnum.Cont,
                    threshold=0.5,
                )
            ],
        ),
        (
            [
                {(0, 0.5): ({(2, 13): (None, None)}, None)},
                {(2, 2): (None, {(1, 4): (None, None)})},
            ],
            [
                MIPDecisionNode(
                    feature_idx=0,
                    threshold=0.5,
                    feature_type=FeatureTypeEnum.Cont,
                    left=MIPDecisionNode(
                        feature_idx=2,
                        threshold=[0, 2, 3],
                        feature_type=FeatureTypeEnum.Cat,
                        left=MIPLeaf(),
                        right=MIPLeaf(),
                    ),
                    right=MIPLeaf(),
                ),
                MIPDecisionNode(
                    feature_idx=2,
                    threshold=[1],
                    feature_type=FeatureTypeEnum.Cat,
                    left=MIPLeaf(),
                    right=MIPDecisionNode(
                        feature_idx=1,
                        threshold=4,
                        feature_type=FeatureTypeEnum.Int,
                        left=MIPLeaf(),
                        right=MIPLeaf(),
                    ),
                ),
            ],
        ),
    ],
)
def test_build_trees_mip_model(
    tree_dicts: list[TreeDict], target_trees_mip: list[MIPDecisionNode]
):
    trees = build_trees_from_dict(tree_dicts, max_depth=3)
    feature_types = jnp.array(
        [FeatureTypeEnum.Cont, FeatureTypeEnum.Int, FeatureTypeEnum.Cat]
    )

    trees_mip = TreesMIPModel(trees=trees, feature_types=feature_types)
    assert trees_mip.trees == target_trees_mip
