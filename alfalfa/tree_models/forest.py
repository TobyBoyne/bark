import torch
import numpy as np
import gpytorch as gpy
from torch.distributions import Normal, Categorical
from typing import Optional, Sequence
from operator import attrgetter
from ..leaf_gp.space import Space
import abc


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
    """
    
    """
    def __init__(self):
        super().__init__()
        self.depth: int = 0
        self.parent: Optional[AlfalfaNode] = None
        self.space: Optional[Space] = None

    def contains_leaves(self, leaves: np.ndarray):
        return np.isin(leaves, self.child_leaves)

    @abc.abstractmethod
    def structure_eq(self, other):
        pass

    @property
    @abc.abstractmethod
    def child_leaves(self):
        return []

    def initialise(self, *args):
        pass

    def get_tree_height(self):
        return 0

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


class DecisionNode(AlfalfaNode):
    def __init__(
        self,
        var_idx=None,
        threshold=None,
        left: Optional["AlfalfaNode"] = None,
        right: Optional["AlfalfaNode"] = None,
    ):
        super().__init__()
        self.var_idx = None if var_idx is None else torch.as_tensor(var_idx)
        self.threshold = None if threshold is None else torch.as_tensor(threshold)
        self.left = LeafNode() if left is None else left
        self.right = LeafNode() if right is None else right


    # Structural methods
    @property
    def child_leaves(self):
        return self.left.child_leaves + self.right.child_leaves

    
    # Model methods
    def initialise(self, space: Space, randomise, depth=0):
        """Sample from the decision node prior.
        
        TODO: This isn't quite the prior!"""
        self.space = space
        self.depth = depth
        self.var_idx = np.random.randint(len(self.space))
        self.threshold = np.random.rand()
        self.left.initialise(space, randomise, depth+1)
        self.right.initialise(space, randomise, depth+1)

    def __call__(self, x):
        var = x[:, self.var_idx]

        if self.var_idx in self.space.cat_idx:
            # categorical - check if value is in subset
            return np.where(
                np.isin(var, self.threshold),
                self.left(x),
                self.right(x)
            )
        else:
            # continuous - check if value is less than threshold
            return np.where(
                var < self.threshold,
                self.left(x),
                self.right(x),
            )

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

    # def extra_repr(self):
    #     if self.var_idx is None or self.threshold is None:
    #         return "(not initialised)"
    #     return f"(x_{self.var_idx}<{self.threshold:.3f})"


    def structure_eq(self, other):
        if isinstance(other, DecisionNode):
            return (
                self.threshold == other.threshold
                and self.var_idx == other.var_idx
                and self.left.structure_eq(other.left)
                and self.right.structure_eq(other.right)
            )
        return False


class AlfalfaTree:
    def __init__(self, height=3, root: Optional[AlfalfaNode] = None):
        if root is not None:
            self.root = root
            # height = root.get_tree_height()
        else:
            self.root = DecisionNode.create_of_height(height)

        self.nodes_by_depth = self._get_nodes_by_depth()
        self.space: Optional[Space] = None

    def initialise(self, space: Space, randomise=True):
        self.space = space
        self.root.initialise(space, randomise)


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
        x1_leaves = self.root(x1)
        x2_leaves = self.root(x2)

        sim_mat = np.equal(x1_leaves[..., :, None], x2_leaves[..., None, :]).astype(float)
        return sim_mat

    def structure_eq(self, other: "AlfalfaTree"):
        return self.root.structure_eq(other.root)


class AlfalfaForest:
    def __init__(
        self, height=None, num_trees=None, trees: Optional[list[AlfalfaTree]] = None
    ):
        self.trees: Sequence[AlfalfaTree]
        if trees:
            self.trees = trees
        else:
            self.trees = [AlfalfaTree(height) for _ in range(num_trees)]

    def initialise(self, space: Space, randomise: bool = True):
        for tree in self.trees:
            tree.initialise(space, randomise)

    def gram_matrix(self, x1: torch.tensor, x2: torch.tensor):
        x1_leaves = np.stack([tree.root(x1) for tree in self.trees], axis=-1)
        x2_leaves = np.stack([tree.root(x2) for tree in self.trees], axis=-1)

        sim_mat = np.equal(x1_leaves[..., :, None, :], x2_leaves[..., None, :, :])
        sim_mat = 1 / len(self.trees) * np.sum(sim_mat, axis=-1)
        return sim_mat

    # def get_extra_state(self):
    #     return {"depth": self.depth, "num_trees": len(self.trees)}

    # def set_extra_state(self, state):
    #     if self.depth != state["depth"]:
    #         raise ValueError(f"Saved model has depth {state['depth']}.")

    #     if len(self.trees) != state["num_trees"]:
    #         raise ValueError(f"Saved model has {state['num_trees']} trees.")

    def structure_eq(self, other: "AlfalfaForest"):
        return all(
            tree.structure_eq(other_tree)
            for tree, other_tree in zip(self.trees, other.trees)
        )
