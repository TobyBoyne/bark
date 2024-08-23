from beartype.typing import Type, Union
from bofire.benchmarks.benchmark import Benchmark

from .constrained import G1, G3, G4, G6, G7, G10, Alkylation
from .mixed import CatAckley, CombinationFunc2, PressureVessel
from .multi_fidelity import CurrinExp2D
from .pest import PestControl
from .unconstrained import (
    Friedman,
    Rastrigin,
    Schwefel,
    StyblinskiTang,
)
from .xgboost_mnist import XGBoostMNIST

BENCHMARK_MAP: dict[str, Type[Benchmark]] = {
    # unconstrained spaces
    "friedman": Friedman,
    "rastrigin": Rastrigin,
    "styblinski_tang": StyblinskiTang,
    "schwefel": Schwefel,
    "combination_func2": CombinationFunc2,
    # constrained spaces
    "g1": G1,
    "g3": G3,
    "g4": G4,
    "g6": G6,
    "g7": G7,
    "g10": G10,
    "alkylation": Alkylation,
    # mixed spaces
    "pest": PestControl,
    "pressure_vessel": PressureVessel,
    "xgboost": XGBoostMNIST,
    # "vae_nas": VAESmall,
    "cat_ackley": CatAckley,
    # multi-fidelity
    # "currin": CurrinExp2D,
}


def map_benchmark(name: str, **kwargs) -> Benchmark:
    """Map a benchmark name to a function that generates the benchmark"""
    return BENCHMARK_MAP[name](**kwargs)
