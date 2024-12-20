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
from .svr_bench import SVRBench
from .welded_beam import WeldedBeam

BENCHMARK_MAP: dict[str, Type[Benchmark]] = {
    # unconstrained spaces
    "Friedman": Friedman,
    "Rastrigin": Rastrigin,
    "StyblinskiTang": StyblinskiTang,
    "Schwefel": Schwefel,
    "CombinationFunc2": CombinationFunc2,
    # constrained spaces
    "G1": G1,
    "G3": G3,
    "G4": G4,
    "G6": G6,
    "G7": G7,
    "G10": G10,
    "Alkylation": Alkylation,
    # mixed spaces
    "PestControl": PestControl,
    "PressureVessel": PressureVessel,
    "XGBoostMNIST": XGBoostMNIST,
    # "vae_nas": VAESmall,
    "CatAckley": CatAckley,
    # multi-fidelity
    # "currin": CurrinExp2D,
    "SVRBench": SVRBench,
    "WeldedBeam": WeldedBeam,
}


def map_benchmark(name: str, **kwargs) -> Benchmark:
    """Map a benchmark name to a function that generates the benchmark"""
    return BENCHMARK_MAP[name](**kwargs)
