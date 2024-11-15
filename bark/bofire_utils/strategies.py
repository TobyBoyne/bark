from bofire.strategies.predictives.predictive import PredictiveStrategy

from bark.bofire_utils.data_models.strategies import TreeKernelStrategy as DataModel


class TreeKernelStrategy(PredictiveStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)

        self.surrogate_specs = data_model.surrogate_specs
