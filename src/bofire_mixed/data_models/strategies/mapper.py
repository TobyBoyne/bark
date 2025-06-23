from bofire.data_models.strategies import api as strategies_data_models
from bofire.strategies.api import map as bofire_map_strategy

from bofire_mixed.data_models.strategies import api as bark_data_models
from bofire_mixed.strategies.bart_grid import BARTGridStrategy
from bofire_mixed.strategies.relaxed_sobo import RelaxedSoboStrategy
from bofire_mixed.strategies.smac import SMACStrategy
from bofire_mixed.strategies.tree_kernel import TreeKernelStrategy

STRATEGY_MAP = {
    bark_data_models.TreeKernelStrategy: TreeKernelStrategy,
    bark_data_models.SMACStrategy: SMACStrategy,
    bark_data_models.RelaxedSoboStrategy: RelaxedSoboStrategy,
    bark_data_models.BARTGridStrategy: BARTGridStrategy,
}


def strategy_map(data_model: strategies_data_models.Strategy) -> TreeKernelStrategy:
    if data_model.__class__ not in STRATEGY_MAP:
        return bofire_map_strategy(data_model)
    cls = STRATEGY_MAP[data_model.__class__]
    return cls(data_model=data_model)
