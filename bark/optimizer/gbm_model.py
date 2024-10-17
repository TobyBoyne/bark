"""Taken from Leaf-GP"""

import collections as coll

import numpy as np

from bark.forest import FeatureTypeEnum


def _build_tree(tree: np.ndarray, feature_types: np.ndarray) -> "GbmNode":
    node_idx = 0
    # TODO: if a tree is a single leaf, can it be skipped entirely?
    if tree[node_idx]["is_leaf"]:
        return None
        # tree = np.zeros_like(tree)
        # tree[0] = (0, 0, -np.inf, 1, 2, 0, 1)
        # tree[1] = (1, 0, 0, 0, 0, 0, 1)
        # tree[2] = (1, 0, 0, 0, 0, 0, 1)

    return GbmNode(tree=tree, node_idx=node_idx, feature_types=feature_types)


class GbmModel:
    """Define a gbm model.

    A `GbmModel` is a object-oriented class to enocode tree-based models from
    different libraries. Supported tree training libraries are: LightGBM.

    Use this class as an interface to read in model data from tree model
    training libraries and translate it to the solver's native structure.

    Parameters
    ----------
    tree_list : list
        Ordered tree list that obeys solver's input specifications

    Attributes
    ----------
    -

    """

    def __init__(self, forest: np.ndarray, feature_types: np.ndarray):
        trees = [_build_tree(tree, feature_types) for tree in forest]
        self.trees = [tree for tree in trees if tree is not None]

        self.n_trees = len(self.trees)

    def get_leaf_encodings(self, tree):
        yield from self.trees[tree].get_leaf_encodings()

    def get_branch_encodings(self, tree):
        yield from self.trees[tree].get_branch_encodings()

    def get_leaf_weight(self, tree, encoding):
        return self.trees[tree].get_leaf_weight(encoding)

    def get_leaf_weights(self, tree):
        return self.trees[tree].get_leaf_weights()

    def get_branch_partition_pair(self, tree, encoding):
        return self.trees[tree].get_branch_partition_pair(encoding)

    def get_left_leaves(self, tree, encoding):
        yield from (encoding + s for s in self.trees[tree].get_left_leaves(encoding))

    def get_right_leaves(self, tree, encoding):
        yield from (encoding + s for s in self.trees[tree].get_right_leaves(encoding))

    def get_branch_partition_pairs(self, tree, leaf_encoding):
        yield from self.trees[tree].get_branch_partition_pairs(leaf_encoding)

    def get_participating_variables(self, tree, leaf):
        return set(self.trees[tree].get_participating_variables(leaf))

    def get_all_participating_variables(self, tree):
        return set(self.trees[tree].get_all_participating_variables())

    def get_var_lower(self, tree, leaf, var, lower):
        return self.trees[tree].get_var_lower(leaf, var, lower)

    def get_var_upper(self, tree, leaf, var, upper):
        return self.trees[tree].get_var_upper(leaf, var, upper)

    def get_var_interval(self, tree, leaf, var, var_interval):
        return self.trees[tree].get_var_interval(leaf, var, var_interval)

    def get_var_break_points(self):
        var_breakpoints = {}
        for tree in self.trees:
            for var, breakpoint in tree.get_all_partition_pairs():
                try:
                    if isinstance(breakpoint, list):
                        var_breakpoints[var].append(breakpoint)
                    else:
                        var_breakpoints[var].add(breakpoint)
                except KeyError:
                    if isinstance(breakpoint, list):
                        var_breakpoints[var] = [breakpoint]
                    else:
                        var_breakpoints[var] = set([breakpoint])

        for k in var_breakpoints.keys():
            if isinstance(var_breakpoints[k], set):
                var_breakpoints[k] = sorted(var_breakpoints[k])

        return var_breakpoints

    def get_leaf_count(self):
        resulting_counter = sum(
            (tree.get_leaf_count() for tree in self.trees), coll.Counter()
        )
        del resulting_counter["leaf"]
        return resulting_counter

    def get_active_leaf_id_vec(self, X):
        leaf_vec = []
        for tree in self.trees:
            tree._get_active_leaf_id_vec(X, leaf_vec)
        return np.asarray(leaf_vec)

    def get_gram_mat(self, X, X2):
        # derive active leaves for x_left and x_right
        x_leaves = np.apply_along_axis(self.get_active_leaf_id_vec, 1, X)
        x2_leaves = np.apply_along_axis(self.get_active_leaf_id_vec, 1, X2)
        sim_mat = np.equal(x_leaves[:, np.newaxis], x2_leaves[np.newaxis, :])
        sim_mat = (1 / self.n_trees) * np.sum(sim_mat, axis=2)
        return sim_mat

    def get_gram_diag(self, X):
        # create diagonal matrix with all ones
        res = np.ones(len(X))
        return res

    def update_var_bounds(self, encodings, var_bnds):
        for tree_id, leaf in encodings:
            self.trees[tree_id]._update_bnds(0, leaf, var_bnds)
        return var_bnds

    def get_active_leaves(self, X):
        all_active_leaves = []
        for tree in self.trees:
            active_leaf = []
            tree._populate_active_leaf_encodings(active_leaf, X)
            all_active_leaves.append("".join(active_leaf))
        return all_active_leaves

    def get_active_area(self, X, cat_idx=None, space=None, volume=False):
        active_splits = {}
        for tree in self.trees:
            tree._populate_active_splits(active_splits, X)

        all_active_splits = {}
        for idx, dim in enumerate(space.dimensions):
            if idx not in cat_idx:
                if idx in active_splits.keys():
                    # if tree splits on this conti var
                    all_active_splits[idx] = active_splits[idx]
                    all_active_splits[idx].insert(0, dim.transformed_bounds[0])
                    all_active_splits[idx].append(dim.transformed_bounds[1])
                else:
                    # add actual bounds of var if tree doesn't split on var
                    all_active_splits[idx] = [
                        dim.transformed_bounds[0],
                        dim.transformed_bounds[1],
                    ]
        # sort all splits and extract modified bounds for vars
        for key in all_active_splits.keys():
            all_active_splits[key] = sorted(list(set(all_active_splits[key])))[:2]

        # return hypervolume if required
        if volume:
            hyper_vol = 1
            for key in all_active_splits.keys():
                hyper_vol *= abs(all_active_splits[key][0] - all_active_splits[key][1])
            return all_active_splits, hyper_vol
        else:
            return all_active_splits

    def get_active_leaf_vars(self, X, model, gbm_label):
        # get active leaves for X
        act_leaves_x = np.asarray([self.get_active_leaves(x) for x in X])

        # generate active_leave_vars
        act_leaf_vars = []
        for data_id, data_enc in enumerate(act_leaves_x):
            temp_lhs = 0
            for tree_id, leaf_enc in enumerate(data_enc):
                temp_lhs += model._z_l[gbm_label, tree_id, leaf_enc]

            temp_lhs *= 1 / len(data_enc)

            act_leaf_vars.append(temp_lhs)

        act_leaf_vars = np.asarray(act_leaf_vars)
        return act_leaf_vars


class GbmType:
    def _populate_active_splits(self, active_splits, X):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                self.left._populate_active_splits(active_splits, X)
            else:
                self.right._populate_active_splits(active_splits, X)
        else:
            if self.split_var != -1:
                if self.split_var not in active_splits.keys():
                    active_splits[self.split_var] = []

                if X[self.split_var] <= self.split_code_pred:
                    self.left._populate_active_splits(active_splits, X)
                    active_splits[self.split_var].append(self.split_code_pred)
                else:
                    self.right._populate_active_splits(active_splits, X)

    def _update_bnds(self, curr_depth, leaf_enc, var_bnds):
        if self.split_var != -1:
            if isinstance(self.split_code_pred, list):
                # categorical variable
                cat_set = set(self.split_code_pred)
                if leaf_enc[curr_depth] == "0":
                    var_bnds[self.split_var] = set(
                        var_bnds[self.split_var]
                    ).intersection(cat_set)
                    self.left._update_bnds(curr_depth + 1, leaf_enc, var_bnds)
                else:
                    var_bnds[self.split_var] = set(var_bnds[self.split_var]).difference(
                        cat_set
                    )
                    self.right._update_bnds(curr_depth + 1, leaf_enc, var_bnds)
            else:
                # continuous variable
                lb, ub = var_bnds[self.split_var]
                if leaf_enc[curr_depth] == "0":
                    ub = min(ub, self.split_code_pred)
                    var_bnds[self.split_var] = (lb, ub)
                    self.left._update_bnds(curr_depth + 1, leaf_enc, var_bnds)
                else:  # if value is '1'
                    lb = max(lb, self.split_code_pred)
                    var_bnds[self.split_var] = (lb, ub)
                    self.right._update_bnds(curr_depth + 1, leaf_enc, var_bnds)

    def _populate_active_leaf_encodings(self, active_leaf, X):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                active_leaf.append("0")
                self.left._populate_active_leaf_encodings(active_leaf, X)
            else:
                active_leaf.append("1")
                self.right._populate_active_leaf_encodings(active_leaf, X)
        else:
            if self.split_var != -1:
                if X[self.split_var] <= self.split_code_pred:
                    active_leaf.append("0")
                    self.left._populate_active_leaf_encodings(active_leaf, X)
                else:
                    active_leaf.append("1")
                    self.right._populate_active_leaf_encodings(active_leaf, X)

    def _get_active_leaf_id_vec(self, X, leaf_vec):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                return self.left._get_active_leaf_id_vec(X, leaf_vec)
            else:
                return self.right._get_active_leaf_id_vec(X, leaf_vec)
        else:
            if X[self.split_var] <= self.split_code_pred:
                return self.left._get_active_leaf_id_vec(X, leaf_vec)
            else:
                return self.right._get_active_leaf_id_vec(X, leaf_vec)


class GbmNode(GbmType):
    """Defines a gbm node which can be a split or leaf.

    Initializing `GbmNode` triggers recursive initialization of all nodes that
    appear in a tree

    Parameters
    ----------
    split_var : int
        Index of split variable used
    split_code_pred : list
        Value that defines split
    tree : list
        List of node dicts that define the tree

    Attributes
    ----------
    split_var : int
        Index of split variable used
    split_code_pred : list
        Value that defines split
    tree : list
        List of node dicts that define the tree
    """

    def __init__(self, tree: np.ndarray, node_idx: int, feature_types: np.ndarray):
        feature_idx = tree[node_idx]["feature_idx"]
        threshold = tree[node_idx]["threshold"]

        self.split_var = feature_idx
        cat_threshold = [
            (int(threshold) >> i) & 1 for i in range(int(threshold).bit_length())
        ]
        self.split_code_pred = (
            threshold
            if feature_types[feature_idx] != FeatureTypeEnum.Cat.value
            else cat_threshold
        )
        # TODO: check categorical features
        child_idx = tree[node_idx]["left"]
        if tree[child_idx]["is_leaf"]:
            self.left = LeafNode(leaf_id=child_idx)
        else:
            self.left = GbmNode(
                tree=tree, node_idx=child_idx, feature_types=feature_types
            )

        # read right node
        child_idx = tree[node_idx]["right"]
        if tree[child_idx]["is_leaf"]:
            self.right = LeafNode(leaf_id=child_idx)
        else:
            self.right = GbmNode(
                tree=tree, node_idx=child_idx, feature_types=feature_types
            )

    def __repr__(self):
        return ", ".join([str(x) for x in [self.split_var, self.split_code_pred]])

    def _get_next_node(self, direction):
        return self.right if int(direction) else self.left

    def get_leaf_encodings(self, current_string=""):
        yield from self.left.get_leaf_encodings(current_string + "0")
        yield from self.right.get_leaf_encodings(current_string + "1")

    def get_branch_encodings(self, current_string=""):
        yield current_string
        yield from self.left.get_branch_encodings(current_string + "0")
        yield from self.right.get_branch_encodings(current_string + "1")

    def get_leaf_weight(self, encoding):
        next_node = self.right if int(encoding[0]) else self.left
        return next_node.get_leaf_weight(encoding[1:])

    def get_leaf_weights(self):
        yield from self.left.get_leaf_weights()
        yield from self.right.get_leaf_weights()

    def get_branch_partition_pair(self, encoding):
        if not encoding:
            return self.split_var, self.split_code_pred
        else:
            next_node = self._get_next_node(encoding[0])
            return next_node.get_branch_partition_pair(encoding[1:])

    def get_left_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_left_leaves(encoding[1:])
        else:
            yield from self.left.get_leaf_encodings("0")

    def get_right_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_right_leaves(encoding[1:])
        else:
            yield from self.right.get_leaf_encodings("1")

    def get_branch_partition_pairs(self, encoding):
        yield (self.split_var, self.split_code_pred)
        try:
            next_node = self.right if int(encoding[0]) else self.left
            next_gen = next_node.get_branch_partition_pairs(encoding[1:])
        except IndexError:
            next_gen = []
        yield from next_gen

    def get_participating_variables(self, encoding):
        yield self.split_var
        next_node = self.right if int(encoding[0]) else self.left
        yield from next_node.get_participating_variables(encoding[1:])

    def get_all_participating_variables(self):
        yield self.split_var
        yield from self.left.get_all_participating_variables()
        yield from self.right.get_all_participating_variables()

    def get_all_partition_pairs(self):
        yield (self.split_var, self.split_code_pred)
        yield from self.left.get_all_partition_pairs()
        yield from self.right.get_all_partition_pairs()

    def get_var_lower(self, encoding, var, lower):
        if encoding:
            if self.split_var == var:
                if int(encoding[0]) == 1:
                    assert lower <= self.split_code_pred
                    lower = self.split_code_pred
            next_node = self.right if int(encoding[0]) else self.left
            return next_node.get_var_lower(encoding[1:], var, lower)
        else:
            return lower

    def get_var_upper(self, encoding, var, upper):
        if encoding:
            if self.split_var == var:
                if int(encoding[0]) == 0:
                    assert upper >= self.split_code_pred
                    upper = self.split_code_pred
            next_node = self.right if int(encoding[0]) else self.left
            return next_node.get_var_upper(encoding[1:], var, upper)
        else:
            return upper

    def get_var_interval(self, encoding, var, var_interval):
        if int(encoding[0]):
            next_node = self.right
            if self.split_var == var:
                var_interval = (self.split_code_pred, var_interval[1])
        else:
            next_node = self.left
            if self.split_var == var:
                var_interval = (var_interval[0], self.split_code_pred)
        return next_node.get_var_interval(encoding[1:], var, var_interval)

    def get_leaf_count(self):
        left_count = self.left.get_leaf_count()
        right_count = self.right.get_leaf_count()
        joint_count = left_count + right_count

        left_key = (self.split_var, self.split_code_pred, 0)
        right_key = (self.split_var, self.split_code_pred, 1)
        joint_count[left_key] += left_count["leaf"]
        joint_count[right_key] += right_count["leaf"]
        return joint_count


class LeafNode(GbmType):
    """Defines a child class of `GbmType`. Leaf nodes have `split_var = -1` and
    `split_code_pred` as leaf value defined by training."""

    def __init__(self, leaf_id):
        self.split_var = -1
        self.split_code_pred = 0.0
        self.leaf_id = leaf_id

    def __repr__(self):
        return ", ".join([str(x) for x in ["LeafNode", self.split_code_pred]])

    def switch_to_maximisation(self):
        """Changes the sign of tree model prediction by changing signs of
        leaf values."""
        self.split_code_pred = -1 * self.split_code_pred

    def get_leaf_encodings(self, current_string=""):
        yield current_string

    def get_branch_encodings(self, current_string=""):
        yield from []

    def get_leaf_weight(self, encoding):
        assert not encoding
        return self.split_code_pred

    def get_leaf_weights(self):
        yield self.split_code_pred

    def get_branch_partition_pair(self, encoding):
        raise Exception("Should not get here.")

    def get_left_leaves(self, encoding):
        raise Exception("Should not get here.")

    def get_right_leaves(self, encoding):
        raise Exception("Should not get here.")

    def get_branch_partition_pairs(self, encoding):
        assert not encoding
        yield from []

    def get_participating_variables(self, encoding):
        assert not encoding
        yield from []

    def get_all_participating_variables(self):
        yield from []

    def get_all_partition_pairs(self):
        yield from []

    def get_var_lower(self, encoding, var, lower):
        assert not encoding
        return lower

    def get_var_upper(self, encoding, var, upper):
        assert not encoding
        return upper

    def get_var_interval(self, encoding, var, var_interval):
        assert not encoding
        return var_interval

    def get_leaf_count(self):
        return coll.Counter({"leaf": 1})

    def _get_active_leaf_id_vec(self, X, leaf_vec):
        return leaf_vec.append(self.leaf_id)
