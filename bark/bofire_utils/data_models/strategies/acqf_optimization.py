from typing import Literal

from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import CategoricalDescriptorInput
from bofire.data_models.strategies.predictives.acqf_optimization import BotorchOptimizer
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
)


class AlternatingBotorchOptimizer(BotorchOptimizer):
    descriptor_method: Literal[CategoricalMethodEnum.FREE] = CategoricalMethodEnum.FREE
    categorical_method: Literal[CategoricalMethodEnum.FREE] = CategoricalMethodEnum.FREE
    discrete_method: Literal[CategoricalMethodEnum.FREE] = CategoricalMethodEnum.FREE
    prefer_exhaustive_search_for_purely_categorical_domains: bool = False

    def validate_surrogate_specs(self, surrogate_specs: BotorchSurrogates):
        # we override the parent since we support FREE
        # we also check that if a categorical with descriptor method is used as one hot encoded the same method is
        # used for the descriptor as for the categoricals
        for m in surrogate_specs.surrogates:
            keys = m.inputs.get_keys(CategoricalDescriptorInput)
            for k in keys:
                input_proc_specs = (
                    m.input_preprocessing_specs[k]
                    if k in m.input_preprocessing_specs
                    else None
                )
                if input_proc_specs == CategoricalEncodingEnum.ONE_HOT:
                    if self.categorical_method != self.descriptor_method:
                        raise ValueError(
                            "One-hot encoded CategoricalDescriptorInput features has to be treated with the same method as categoricals.",
                        )
