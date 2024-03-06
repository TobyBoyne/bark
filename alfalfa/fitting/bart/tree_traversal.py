from beartype.typing import Callable, Union

from ...forest import AlfalfaTree, DecisionNode, LeafNode


def in_order_conditional(
    tree: AlfalfaTree,
    condition: Callable[[Union[DecisionNode, LeafNode]], bool],
    node: Union[DecisionNode, LeafNode, None] = None,
):
    if node is None:
        node = tree.root

    if isinstance(node, DecisionNode):
        yield from in_order_conditional(tree, condition, node.left)

    if condition(node):
        yield node

    if isinstance(node, DecisionNode):
        yield from in_order_conditional(tree, condition, node.right)


def terminal_nodes(tree: AlfalfaTree) -> list[LeafNode]:
    """Find all leaves"""

    def cond(node):
        return isinstance(node, LeafNode)

    return list(in_order_conditional(tree, cond))


def singly_internal_nodes(tree: AlfalfaTree) -> list[DecisionNode]:
    """Find all decision nodes where both children are leaves"""

    def cond(node):
        return (
            isinstance(node, DecisionNode)
            and isinstance(node.left, LeafNode)
            and isinstance(node.right, LeafNode)
        )

    return list(in_order_conditional(tree, cond))
