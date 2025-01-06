from typing import Literal

from bofire.data_models.domain.api import Domain
from pydantic import BaseModel


class RelaxedSoboStrategy(BaseModel):
    type: Literal["RelaxedSoboStrategy"] = "RelaxedSoboStrategy"
    domain: Domain
    seed: int
