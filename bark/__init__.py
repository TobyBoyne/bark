# register the alternating mixed optimizer
# this needs to be here, to guarantee it is run when BARK is being used
from bofire.strategies.predictives.acqf_optimization import OPTIMIZER_MAP

from bark.bofire_utils.data_models.strategies.acqf_optimization import (
    AlternatingBotorchOptimizer as AlternatingBotorchOptimizerDataModel,
)
from bark.bofire_utils.strategies.acqf_optimization import AlternatingBotorchOptimizer

OPTIMIZER_MAP[AlternatingBotorchOptimizerDataModel] = AlternatingBotorchOptimizer
