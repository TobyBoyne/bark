from typing import Literal

from bofire.data_models.acquisition_functions.api import (
    AnySingleObjectiveAcquisitionFunction,
    qLogNEI,
)
from bofire.data_models.domain.api import Domain
from pydantic import BaseModel, Field


class RelaxedSoboStrategy(BaseModel):
    type: Literal["RelaxedSoboStrategy"] = "RelaxedSoboStrategy"
    domain: Domain
    seed: int
    acquisition_function: AnySingleObjectiveAcquisitionFunction = Field(
        default_factory=lambda: qLogNEI(),
    )
