# register the alternating mixed optimizer
from bofire.strategies.predictives.acqf_optimization import OPTIMIZER_MAP

from bark.bofire_utils.data_models.strategies.acqf_optimization import (
    AlternatingBotorchOptimizer as AlternatingBotorchOptimizerDataModel,
)
from bark.bofire_utils.strategies.acqf_optimization import AlternatingBotorchOptimizer

OPTIMIZER_MAP[AlternatingBotorchOptimizerDataModel] = AlternatingBotorchOptimizer
