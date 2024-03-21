import abc

import numpy as np
from beartype.typing import Callable, Optional, Sequence

from .utils.space import Space

InitFuncType = Optional[Callable[["DecisionNode"], None]]


def _leaf_id_iter():
    i = 0
    while True:
        yield i
        i += 1


_leaf_id = _leaf_id_iter()


# def prune_tree_hook(module, incompatible_keys):
#     """Post hook for load_state_dict to handle missing nodes.

#     This transforms any nodes that are missing data to leaves, effectively
#     'pruning' branches of the tree. This function is to be used as a pre-hook
#     for torch.load_state_dict, must be registered before loading the data."""

#     while incompatible_keys.missing_keys:
#         key = incompatible_keys.missing_keys.pop()
#         *parent_key, child, _ = key.split(".")
#         parent_node = attrgetter(".".join(parent_key))(module)
#         setattr(parent_node, child, LeafNode())


class AlfalfaNode:
    """ """

    def __init__(self):
        super().__init__()
        self.depth: int = 0
        self.parent: Optional[tuple[DecisionNode, str]] = None
        self.tree: Optional[AlfalfaTree] = None

    def contains_leaves(self, leaves: np.ndarray):
        return np.isin(leaves, self.child_leaves)

    @abc.abstractmethod
    def structure_eq(self, other):
        pass

    @property
    @abc.abstractmethod
    def child_leaves(self):
        return []

    @property
    def space(self):
        return self.tree.space

    def initialise(self, depth, *args):
        self.depth = depth

    def get_tree_height(self):
        return 0

    def replace_self(self, new_node: "AlfalfaNode"):
        """Replace this node with a different node.

        Assign either parent_node.left or parent_node.right to the new node,
        depending on the direction from the current node to the parent"""
        if self.parent is None:
            # this is the root of the tree
            new_node.tree = self.tree
            self.tree.root = new_node
            new_node.depth = 0

            self.tree = None
            return

        parent_node, child_direction = self.parent
        setattr(parent_node, child_direction, new_node)

        # detach self from the tree
        self.depth = 0
        self.parent = None
        self.tree = None

    def as_dict(self):
        return {}

    @classmethod
    def from_dict(cls, d):
        return LeafNode()


class LeafNode(AlfalfaNode):
    def __init__(self):
        super().__init__()
        self.leaf_id = next(_leaf_id)

    @property
    def child_leaves(self):
        return [self.leaf_id]

    def __call__(self, _x):
        return self.leaf_id

    def structure_eq(self, other):
        return isinstance(other, LeafNode)

    def __repr__(self):
        return "L"


class DecisionNode(AlfalfaNode):
    def __init__(
        self,
        var_idx=None,
        threshold=None,
        left: Optional["AlfalfaNode"] = None,
        right: Optional["AlfalfaNode"] = None,
    ):
        super().__init__()
        self.var_idx = None if var_idx is None else var_idx
        self.threshold = None if threshold is None else threshold

        self._left: AlfalfaNode = None
        self._right: AlfalfaNode = None

        self.left = LeafNode() if left is None else left
        self.right = LeafNode() if right is None else right

    # Structural methods
    @property
    def child_leaves(self):
        return self.left.child_leaves + self.right.child_leaves

    def _set_child_data(self, child: AlfalfaNode, recurse=True):
        """Set the metadata (depth, parent, etc) for a child node"""
        child.tree = self.tree
        child.depth = self.depth + 1
        if recurse and isinstance(child, DecisionNode):
            child._set_child_data(child.left, recurse=True)
            child._set_child_data(child.right, recurse=True)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node: AlfalfaNode):
        self._left = node
        self._left.parent = (self, "left")
        self._set_child_data(node, recurse=True)

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node: AlfalfaNode):
        self._right = node
        self._right.parent = (self, "right")
        self._set_child_data(node, recurse=True)

    # Model methods
    def initialise(self, depth, init_func: InitFuncType):
        """Sample from the decision node prior."""
        super().initialise(depth)
        if init_func is not None:
            init_func(self)
        self.left.initialise(depth + 1, init_func)
        self.right.initialise(depth + 1, init_func)

    def __call__(self, x: np.ndarray, allow_not_initialised=False):
        if self.threshold is None and allow_not_initialised:
            # raise ValueError("This node is not initialised.")
            return np.full((x.shape[0],), self.left(x))

        var = x[:, self.var_idx]

        if self.var_idx in self.space.cat_idx:
            # categorical - check if value is in subset
            return np.where(np.isin(var, self.threshold), self.left(x), self.right(x))
        else:
            # continuous - check if value is less than threshold
            return np.where(
                var <= self.threshold,
                self.left(x),
                self.right(x),
            )

    def __repr__(self):
        return f"N{self.var_idx}({self.left}), ({self.right})"

    @classmethod
    def create_of_height(cls, d):
        """Create a node with depth d"""
        if d == 0:
            return LeafNode()
        else:
            left = cls.create_of_height(d - 1)
            right = cls.create_of_height(d - 1)
            return DecisionNode(left=left, right=right)

    def get_tree_height(self):
        return 1 + max(self.left.get_tree_height(), self.right.get_tree_height())

    def structure_eq(self, other):
        if isinstance(other, DecisionNode):
            return (
                self.threshold == other.threshold
                and self.var_idx == other.var_idx
                and self.left.structure_eq(other.left)
                and self.right.structure_eq(other.right)
            )
        return False

    def as_dict(self):
        return {
            "var_idx": self.var_idx,
            "threshold": self.threshold,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        if not d:
            return super().from_dict(d)
        return cls(
            var_idx=d["var_idx"],
            threshold=d["threshold"],
            left=DecisionNode.from_dict(d["left"]),
            right=DecisionNode.from_dict(d["right"]),
        )


class AlfalfaTree:
    def __init__(self, height=3, root: Optional[AlfalfaNode] = None):
        if root is not None:
            self.root = root
        else:
            self.root = DecisionNode.create_of_height(height)

        self.nodes_by_depth = self._get_nodes_by_depth()
        self.root.tree = self
        # propagate self.tree throughout tree
        if isinstance(self.root, DecisionNode):
            self.root._set_child_data(self.root.left)
            self.root._set_child_data(self.root.right)

        self.space: Optional[Space] = None

    def initialise(self, space: Space, init_func: InitFuncType = None):
        self.space = space
        self.root.initialise(0, init_func)

    def _get_nodes_by_depth(self) -> dict[int, list[AlfalfaNode]]:
        nodes = [self.root]
        nodes_by_depth = {}
        depth = 0
        while nodes:
            nodes_by_depth[depth] = [*nodes]
            new_nodes = []
            for node in nodes:
                if isinstance(node, DecisionNode):
                    new_nodes += [node.left, node.right]
            nodes = new_nodes
            depth += 1
        return nodes_by_depth

    def gram_matrix(self, x1: np.ndarray, x2: np.ndarray):
        if isinstance(self.root, LeafNode):
            return np.ones((x1.shape[-2], x2.shape[-2]), dtype=float)
        x1_leaves = self(x1)
        x2_leaves = self(x2)

        sim_mat = np.equal(x1_leaves[..., :, None], x2_leaves[..., None, :]).astype(
            float
        )
        return sim_mat

    def get_leaf_vectors(self, x: np.ndarray):
        x_leaves = self(x)
        all_leaves = np.array(self.root.child_leaves)
        return (np.equal(x_leaves[:, None], all_leaves[None, :])).astype(float)

    def __call__(self, x):
        if isinstance(self.root, DecisionNode):
            return self.root(x)
        else:
            return np.full((x.shape[0],), fill_value=self.root.leaf_id)

    def structure_eq(self, other: "AlfalfaTree"):
        return self.root.structure_eq(other.root)

    def as_dict(self):
        return {"tree_model_type": "tree", "root": self.root.as_dict()}

    @classmethod
    def from_dict(cls, d):
        root = DecisionNode.from_dict(d["root"])
        return cls(root=root)


class AlfalfaForest:
    def __init__(
        self, height=None, num_trees=None, trees: Optional[list[AlfalfaTree]] = None
    ):
        self.trees: Sequence[AlfalfaTree]
        if trees:
            self.trees = trees
        else:
            self.trees = [AlfalfaTree(height) for _ in range(num_trees)]

    def initialise(self, space: Space, init_func: InitFuncType = None):
        self.space = space
        for tree in self.trees:
            tree.initialise(space, init_func)

    def gram_matrix(self, x1: np.ndarray, x2: np.ndarray):
        x1_leaves = np.stack([tree(x1) for tree in self.trees], axis=-1)
        x2_leaves = np.stack([tree(x2) for tree in self.trees], axis=-1)

        sim_mat = np.equal(x1_leaves[..., :, None, :], x2_leaves[..., None, :, :])
        sim_mat = 1 / len(self.trees) * np.sum(sim_mat, axis=-1)
        return sim_mat

    def structure_eq(self, other: "AlfalfaForest"):
        return all(
            tree.structure_eq(other_tree)
            for tree, other_tree in zip(self.trees, other.trees)
        )

    def as_dict(self):
        return {
            "tree_model_type": "forest",
            "trees": [tree.as_dict() for tree in self.trees],
        }

    @classmethod
    def from_dict(cls, d):
        trees = [AlfalfaTree.from_dict(d_tree) for d_tree in d["trees"]]
        return cls(trees=trees)
