from ...tree_models.forest import AlfalfaTree, DecisionNode, LeafNode
from typing import Callable, Union

def in_order_conditional(
        tree: AlfalfaTree, 
        condition: Callable[[Union[DecisionNode, LeafNode]], bool], 
        node: Union[DecisionNode, LeafNode, None] = None):
    
    if node is None:
        node = tree.root

    if isinstance(node, DecisionNode):
        yield from in_order_conditional(tree, condition, node.left)

    if condition(node):
        yield node

    if isinstance(node, DecisionNode):
        yield from in_order_conditional(tree, condition, node.right)

def terminal_nodes(tree: AlfalfaTree):
    """Find all leaves"""
    def cond(node):
        return isinstance(node, LeafNode)
    
    return list(in_order_conditional(tree, cond))

def singly_internal_nodes(tree: AlfalfaTree):
    """Find all decision nodes where both children are leaves"""
    def cond(node):
        return (
            isinstance(node, DecisionNode) and 
            isinstance(node.left, LeafNode) and 
            isinstance(node.right, LeafNode)
        )
    
    return list(in_order_conditional(tree, cond))


def assign_node_depth(tree: AlfalfaTree):
    def node_depth(node: DecisionNode, depth):
        node.depth = depth
        if isinstance(node, DecisionNode):
            node_depth(node.left, depth+1)
            node_depth(node.right, depth+1)

    node_depth(tree.root, 0)
    