from bofire.data_models.strategies import api as strategies_data_models
from bofire.strategies.api import map as bofire_map_strategy

from bark.bofire_utils.data_models.strategies import api as bark_data_models
from bark.bofire_utils.strategies.tree_kernel import TreeKernelStrategy

STRATEGY_MAP = {
    bark_data_models.TreeKernelStrategy: TreeKernelStrategy,
}


def strategy_map(data_model: strategies_data_models.Strategy) -> TreeKernelStrategy:
    if data_model.__class__ not in STRATEGY_MAP:
        return bofire_map_strategy(data_model)
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
