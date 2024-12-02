from bofire.data_models.surrogates import api as data_models
from bofire.surrogates.surrogate import Surrogate

from bark.bofire_utils.data_models import surrogates as bark_data_models
from bark.bofire_utils.surrogates import BARKSurrogate, LeafGPSurrogate

SURROGATE_MAP: dict[type[data_models.Surrogate], type[Surrogate]] = {
    bark_data_models.LeafGPSurrogate: LeafGPSurrogate,
    bark_data_models.BARKSurrogate: BARKSurrogate,
}


def surrogate_map(data_model: data_models.Surrogate) -> BARKSurrogate | LeafGPSurrogate:
    cls = SURROGATE_MAP[data_model.__class__]
    return cls.from_spec(data_model=data_model)
