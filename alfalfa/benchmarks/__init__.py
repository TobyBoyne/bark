from beartype.typing import Union

from .base import CatSynFunc, SynFunc
from .constrained import G1, G3, G4, G6, G7, G10, Alkylation
from .mixed import PressureVessel, VAESmall
from .multi_fidelity import CurrinExp2D
from .unconstrained import (
    Branin,
    Hartmann6D,
    Himmelblau1D,
    Rastrigin,
    Schwefel,
    StyblinskiTang,
)

BENCHMARK_MAP = {
    # unconstrained spaces
    "hartmann6d": Hartmann6D,
    "himmelblau1d": Himmelblau1D,
    "branin": Branin,
    "rastrigin": Rastrigin,
    "styblinski_tang": StyblinskiTang,
    "schwefel": Schwefel,
    # constrained spaces
    "g1": G1,
    "g3": G3,
    "g4": G4,
    "g6": G6,
    "g7": G7,
    "g10": G10,
    "alkylation": Alkylation,
    # mixed spaces
    "pressure_vessel": PressureVessel,
    "vae_nas": VAESmall,
    # multi-fidelity
    "currin": CurrinExp2D,
}


def map_benchmark(name: str, **kwargs) -> Union[SynFunc, CatSynFunc]:
    """Map a benchmark name to a function that generates the benchmark

    Args:
        name (str): the name of the benchmark
        **kwargs: additional arguments to pass to the benchmark function

    Returns:
        Union[SynFunc, CatSynFunc]: the benchmark function
    """
    return BENCHMARK_MAP[name](**kwargs)
