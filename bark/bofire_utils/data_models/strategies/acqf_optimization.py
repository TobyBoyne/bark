from typing import Literal

from bofire.data_models.enum import CategoricalMethodEnum
from bofire.data_models.strategies.predictives.acqf_optimization import BotorchOptimizer


class AlternatingBotorchOptimizer(BotorchOptimizer):
    descriptor_method: Literal[CategoricalMethodEnum.FREE] = CategoricalMethodEnum.FREE
    categorical_method: Literal[CategoricalMethodEnum.FREE] = CategoricalMethodEnum.FREE
    discrete_method: Literal[CategoricalMethodEnum.FREE] = CategoricalMethodEnum.FREE
