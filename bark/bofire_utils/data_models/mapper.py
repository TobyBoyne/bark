from bofire.data_models.strategies import api as strategies_data_models
from bofire.data_models.surrogates import api as surrogates_data_models
from bofire.strategies.api import map as bofire_map_strategy
from bofire.surrogates.api import map as bofire_map_surrogate
from bofire.surrogates.surrogate import Surrogate

from bark.bofire_utils.data_models import api as bark_data_models
from bark.bofire_utils.strategies.tree_kernel import TreeKernelStrategy
from bark.bofire_utils.surrogates.bark import BARKPriorSurrogate, BARKSurrogate
from bark.bofire_utils.surrogates.leafgp import LeafGPSurrogate

SURROGATE_MAP: dict[type[surrogates_data_models.Surrogate], type[Surrogate]] = {
    bark_data_models.LeafGPSurrogate: LeafGPSurrogate,
    bark_data_models.BARKSurrogate: BARKSurrogate,
    bark_data_models.BARKPriorSurrogate: BARKPriorSurrogate,
}

STRATEGY_MAP = {
    bark_data_models.TreeKernelStrategy: TreeKernelStrategy,
}


def surrogate_map(
    data_model: surrogates_data_models.Surrogate,
) -> BARKSurrogate | LeafGPSurrogate:
    if data_model.__class__ not in SURROGATE_MAP:
        return bofire_map_surrogate(data_model)
    cls = SURROGATE_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)


def strategy_map(data_model: strategies_data_models.Strategy) -> TreeKernelStrategy:
    if data_model.__class__ not in STRATEGY_MAP:
        return bofire_map_strategy(data_model)
    cls = STRATEGY_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
