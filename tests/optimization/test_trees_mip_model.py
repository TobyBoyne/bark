import jax.numpy as jnp
import pytest

from bark.enums import FeatureTypeEnum
from bark.optimizer.mip_model import MIPDecisionNode, MIPLeaf, TreesMIPModel
from bark.testing.build_test_trees import TreeDict, build_trees_from_dict

FEATURE_TYPES = jnp.array(
    [FeatureTypeEnum.Cont, FeatureTypeEnum.Int, FeatureTypeEnum.Cat]
)

trees_mip_testcase_1 = [
    MIPDecisionNode(
        left=MIPLeaf(),
        right=MIPLeaf(),
        feature_idx=0,
        feature_type=FeatureTypeEnum.Cont,
        threshold=0.5,
    )
]

trees_mip_testcase_2 = [
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
]


@pytest.mark.parametrize(
    argnames=("target_trees_mip", "tree_dicts"),
    argvalues=[
        (
            trees_mip_testcase_1,
            [
                None,
                {(0, 0.5): (None, None)},
                None,
            ],
        ),
        (
            trees_mip_testcase_2,
            [
                {(0, 0.5): ({(2, 13): (None, None)}, None)},
                {(2, 2): (None, {(1, 4): (None, None)})},
            ],
        ),
    ],
)
def test_build_trees_mip_model(
    target_trees_mip: list[MIPDecisionNode],
    tree_dicts: list[TreeDict],
):
    trees = build_trees_from_dict(tree_dicts, max_depth=3)

    trees_mip = TreesMIPModel(trees=trees, feature_types=FEATURE_TYPES)
    assert trees_mip.trees == target_trees_mip


@pytest.mark.parametrize(
    ("target_trees_mip", "left_leaves", "right_leaves"),
    [
        (trees_mip_testcase_1, [["0"]], [["1"]]),
        (trees_mip_testcase_2, [["00", "01"], ["0"]], [["1"], ["10", "11"]]),
    ],
)
def test_get_child_leaves(
    target_trees_mip: list[MIPDecisionNode],
    left_leaves: list[list[str]],
    right_leaves: list[list[str]],
):
    for tree, l_lf, r_lf in zip(target_trees_mip, left_leaves, right_leaves):
        assert list(tree.get_left_leaves("")) == l_lf
        assert list(tree.get_right_leaves("")) == r_lf
