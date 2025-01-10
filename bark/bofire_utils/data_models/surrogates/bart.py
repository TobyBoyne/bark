from typing import Literal, Type

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    AnyOutput,
    CategoricalInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.surrogates.trainable import TrainableSurrogate
from pydantic import field_validator


class BARTSurrogate(Surrogate, TrainableSurrogate):
    type: Literal["BARKSurrogate"] = "BARKSurrogate"

    seed: int = 0

    @field_validator("input_preprocessing_specs")
    @classmethod
    def validate_input_preprocessing_specs(cls, v, info):
        # when validator for inputs fails, this validator is still checked and causes an Exception error instead of a ValueError
        # fix this by checking if inputs is in info.data
        if "inputs" not in info.data:
            return None

        inputs: Inputs = info.data["inputs"]
        categorical_keys = inputs.get_keys(CategoricalInput, exact=True)
        for key in categorical_keys:
            if (
                v.get(key, CategoricalEncodingEnum.ORDINAL)
                != CategoricalEncodingEnum.ORDINAL
            ):
                raise ValueError(
                    "BART based models have to use ordinal encoding for categoricals",
                )
            v[key] = CategoricalEncodingEnum.ORDINAL
        return v

    @classmethod
    def is_output_implemented(cls, my_type: Type[AnyOutput]) -> bool:
        """Method to check output type for surrogate models
        Args:
            my_type: continuous or categorical output
        Returns:
            bool: True if the output type is valid for the surrogate chosen, False otherwise
        """
        return isinstance(my_type, type(ContinuousOutput))
