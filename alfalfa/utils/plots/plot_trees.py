import matplotlib.pyplot as plt
import numpy as np
from beartype.typing import Optional
from matplotlib.patches import Circle

from ...forest import AlfalfaForest, AlfalfaNode, AlfalfaTree, DecisionNode

NODE_SIZE = 0.5
LEFT = np.array([-1.0, -2.0])
RIGHT = np.array([1.0, -2.0])
TREE_SPACING = np.array([5.0, 0.0])


def plot_tree(tree: AlfalfaTree, ax: plt.Axes, xy: Optional[np.ndarray] = None):
    def plot_node(node: AlfalfaNode, xy: np.ndarray):
        ax.add_patch(Circle(xy, NODE_SIZE))
        if isinstance(node, DecisionNode):
            ax.plot(*zip(xy, xy + LEFT), color="black", linewidth=3)
            ax.plot(*zip(xy, xy + RIGHT), color="black", linewidth=3)
            plot_node(node.left, xy + LEFT)
            plot_node(node.right, xy + RIGHT)

    if xy is None:
        xy = np.array([0.0, 0.0])
    plot_node(tree.root, np.asarray(xy))


def plot_forest(forest: AlfalfaForest, ax: plt.Axes):
    xy = np.array([0.0, 0.0])
    for i, tree in enumerate(forest.trees):
        plot_tree(tree, ax, xy + i * TREE_SPACING)
