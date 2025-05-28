from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDataModel
from bofire.strategies.api import SoboStrategy

from bark.benchmarks.mixed import DiscreteAckley
from bark.bofire_utils.data_models.strategies.acqf_optimization import (
    AlternatingBotorchOptimizer,
)

benchmark = DiscreteAckley()

dm = SoboStrategyDataModel(
    domain=benchmark.domain, acquisition_optimizer=AlternatingBotorchOptimizer()
)

strategy = SoboStrategy(data_model=dm)

samples = benchmark.domain.inputs.sample(10)
experiments = benchmark.f(samples, return_complete=True)

strategy.tell(experiments)
print(strategy.ask(1))
