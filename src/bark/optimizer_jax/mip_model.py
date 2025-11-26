import collections as coll
from dataclasses import dataclass
from typing import Generator, Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from bark import forest, types
from bark.enums import FeatureTypeEnum, NodeState
from bark.types.optimizer import GurobiOptimizerModel
from bark.utils.bit_operations import next_power_of_2_exponent


def _binary_mask_threshold_to_list(threshold: types.IndexT) -> list[int]:
    return [
        i
        for i in range(next_power_of_2_exponent(int(threshold)))
        if (int(threshold) >> i) & 1
    ]


def _build_tree(
    tree: types.Tree, feature_types: types.FeatTypesT, node_idx: types.IndexT = 0
) -> "MIPDecisionNode | MIPLeaf":
    if tree.feature_idx[node_idx] == NodeState.Leaf:
        return MIPLeaf()

    left_child = _build_tree(tree, feature_types, node_idx=forest.left(node_idx))
    right_child = _build_tree(tree, feature_types, node_idx=forest.right(node_idx))
    feature_idx = int(tree.feature_idx[node_idx])
    threshold = int(tree.threshold[node_idx])
    feature_type = FeatureTypeEnum(int(feature_types[feature_idx]))

    if feature_type == FeatureTypeEnum.Cat:
        threshold = _binary_mask_threshold_to_list(threshold)

    return MIPDecisionNode(
        left=left_child,
        right=right_child,
        feature_idx=feature_idx,
        threshold=threshold,
        feature_type=feature_type,
    )


class MIPModel:
    def __init__(self, trees: types.Tree, feature_types: types.FeatTypesT):
        mip_trees = [_build_tree(tree, feature_types) for tree in trees]
        self.trees = [tree for tree in mip_trees if not isinstance(tree, MIPLeaf)]

        self.n_trees = len(self.trees)

    def get_leaf_encodings(self, tree: int):
        yield from self.trees[tree].get_leaf_encodings()

    def get_branch_encodings(self, tree: int):
        yield from self.trees[tree].get_branch_encodings()

    def get_branch_partition_pair(self, tree: int, encoding):
        return self.trees[tree].get_branch_partition_pair(encoding)

    def get_left_leaves(self, tree: int, encoding: str):
        yield from (encoding + s for s in self.trees[tree].get_left_leaves(encoding))

    def get_right_leaves(self, tree: int, encoding: str):
        yield from (encoding + s for s in self.trees[tree].get_right_leaves(encoding))

    def get_var_break_points(self) -> dict[int, list[float] | list[list[int]]]:
        cont_breakpoints = coll.defaultdict[int, set[float]](set)
        cat_breakpoints = coll.defaultdict[int, list[list[int]]](list)
        for tree in self.trees:
            for var, breakpoint in tree.get_all_partition_pairs():
                if isinstance(breakpoint, list):
                    # node is categorical
                    cat_breakpoints[var].append(breakpoint)
                else:
                    cont_breakpoints[var].add(breakpoint)

        return cat_breakpoints | {k: sorted(v) for k, v in cont_breakpoints.items()}

    def update_var_bounds_inplace(
        self,
        encodings: list[tuple[int, str]],
        var_bnds: list[tuple[float, float] | list[int]],
    ) -> None:
        for tree_id, leaf in encodings:
            self.trees[tree_id].update_var_bounds_inplace(0, leaf, var_bnds)

    def get_active_leaves(self, X: Float[Array, "N d"]) -> list[str]:
        all_active_leaves: list[str] = []
        for tree in self.trees:
            active_leaf = []
            tree.update_active_leaf_inplace(active_leaf, X)
            all_active_leaves.append("".join(active_leaf))
        return all_active_leaves

    def get_active_leaf_vars(
        self, X: Float[Array, "N d"], model: GurobiOptimizerModel, gbm_label: str
    ) -> Float[Array, " N"]:
        # get active leaves for X
        act_leaves_x = [self.get_active_leaves(x) for x in X]

        # generate active_leave_vars
        act_leaf_vars: list[float] = []
        for data_enc in act_leaves_x:
            temp_lhs = 0
            for tree_id, leaf_enc in enumerate(data_enc):
                temp_lhs += model._z_l[gbm_label, tree_id, leaf_enc]

            temp_lhs *= 1 / len(data_enc)

            act_leaf_vars.append(temp_lhs)

        return jnp.asarray(act_leaf_vars)


@dataclass
class MIPDecisionNode:
    left: "MIPDecisionNode | MIPLeaf"
    right: "MIPDecisionNode | MIPLeaf"

    feature_idx: int
    feature_type: FeatureTypeEnum
    threshold: float | list[int]

    def _get_next_node(self, direction: str):
        return self.right if int(direction) else self.left

    def get_leaf_encodings(self, current_string="") -> Generator[str]:
        yield from self.left.get_leaf_encodings(current_string + "0")
        yield from self.right.get_leaf_encodings(current_string + "1")

    def get_branch_encodings(self, current_string="") -> Generator[str]:
        yield current_string
        yield from self.left.get_branch_encodings(current_string + "0")
        yield from self.right.get_branch_encodings(current_string + "1")

    def get_branch_partition_pair(self, encoding: str):
        if not encoding:
            return self.feature_idx, self.threshold
        else:
            next_node = self._get_next_node(encoding[0])
            if isinstance(next_node, MIPLeaf):
                raise ValueError("Cannot get partition pair from leaf.")
            return next_node.get_branch_partition_pair(encoding[1:])

    def get_all_partition_pairs(self) -> Generator[tuple[int, float | list[int]]]:
        yield (self.feature_idx, self.threshold)
        yield from self.left.get_all_partition_pairs()
        yield from self.right.get_all_partition_pairs()

    def _get_child_leaves(
        self, encoding: str, direction: Literal["0", "1"]
    ) -> Generator[str]:
        if encoding:
            next_node = self._get_next_node(encoding[0])
            if isinstance(next_node, MIPLeaf):
                raise ValueError("Cannot get leaves from leaf.")
            yield from next_node._get_child_leaves(encoding[1:], direction)
        else:
            yield from self.left.get_leaf_encodings(direction)

    def get_left_leaves(self, encoding: str):
        return self._get_child_leaves(encoding, direction="0")

    def get_right_leaves(self, encoding: str):
        return self._get_child_leaves(encoding, direction="1")

    def update_var_bounds_inplace(
        self,
        curr_depth: int,
        leaf_enc: str,
        var_bnds: list[tuple[float, float] | list[int]],
    ) -> None:
        direction = leaf_enc[curr_depth]

        if self.feature_type == FeatureTypeEnum.Cat:
            assert isinstance(self.threshold, list)
            cat_set = set(self.threshold)

            if direction == "0":
                var_bnds[self.feature_idx] = list(  # type: ignore
                    set(var_bnds[self.feature_idx]).intersection(cat_set)
                )
            else:
                var_bnds[self.feature_idx] = list(  # type: ignore
                    set(var_bnds[self.feature_idx]).difference(cat_set)
                )

        else:
            assert isinstance(self.threshold, float)
            lb, ub = var_bnds[self.feature_idx]
            if direction == "0":
                ub = min(ub, self.threshold)
            else:
                lb = max(lb, self.threshold)
            var_bnds[self.feature_idx] = (lb, ub)

        child_node = self._get_next_node(direction)
        if isinstance(child_node, MIPDecisionNode):
            child_node.update_var_bounds_inplace(curr_depth + 1, leaf_enc, var_bnds)

    def update_active_leaf_inplace(
        self, active_leaf: list[str], X: Float[Array, " d"]
    ) -> None:
        if self.feature_type == FeatureTypeEnum.Cat:
            assert isinstance(self.threshold, list)
            direction = "0" if X[self.feature_idx] in self.threshold else "1"
        else:
            assert isinstance(self.threshold, float)
            direction = "0" if X[self.feature_idx] <= self.threshold else "1"

        active_leaf.append(direction)
        child_node = self._get_next_node(direction)
        if isinstance(child_node, MIPDecisionNode):
            child_node.update_active_leaf_inplace(active_leaf, X)


@dataclass
class MIPLeaf:
    def get_leaf_encodings(self, current_string=""):
        yield current_string

    def get_branch_encodings(self, current_string=""):
        yield from []

    def get_all_partition_pairs(self):
        yield from []
