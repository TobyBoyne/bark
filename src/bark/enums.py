from enum import IntEnum

MAX_DEPTH = 6


class FeatureTypeEnum(IntEnum):
    Cat = 0
    Int = 1
    Cont = 2


class TreeProposalEnum(IntEnum):
    Grow = 0
    Prune = 1
    Change = 2


class NodeState(IntEnum):
    """Special states that feature index nodes can take."""

    Leaf = -1
    Inactive = -2
