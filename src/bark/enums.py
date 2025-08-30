from enum import Enum


class FeatureTypeEnum(Enum):
    Cat = 0
    Int = 1
    Cont = 2


class TreeProposalEnum(Enum):
    Grow = 0
    Prune = 1
    Change = 2
