from beartype.typing import Type, Union
from bofire.benchmarks.api import Hartmann
from bofire.benchmarks.benchmark import Benchmark

from .cco.cellular_network import CCOBench
from .constrained import G1, G3, G4, G6, G7, G10, Alkylation
from .dataset import DatasetBenchmark
from .MAX_bandit import MAXBandit
from .mixed import CombinationFunc2, DiscreteAckley, DiscreteRosenbrock, PressureVessel
from .multi_fidelity import CurrinExp2D
from .pest import PestControl
from .reaction_emulator import ToyReaction
from .svr_bench import SVRBench
from .tree_function import TreeFunction
from .unconstrained import (
    Friedman,
    Rastrigin,
    Schwefel,
    StyblinskiTang,
)
from .xgboost_mnist import XGBoostMNIST

BENCHMARK_MAP: dict[str, Type[Benchmark]] = {
    # unconstrained spaces
    "Friedman": Friedman,
    "Rastrigin": Rastrigin,
    "StyblinskiTang": StyblinskiTang,
    "Schwefel": Schwefel,
    "CombinationFunc2": CombinationFunc2,
    "Hartmann": Hartmann,
    # constrained spaces
    "G1": G1,
    "G3": G3,
    "G4": G4,
    "G6": G6,
    "G7": G7,
    "G10": G10,
    "Alkylation": Alkylation,
    # mixed spaces
    "ToyReaction": ToyReaction,
    "PestControl": PestControl,
    "PressureVessel": PressureVessel,
    "XGBoostMNIST": XGBoostMNIST,
    # "vae_nas": VAESmall,
    "DiscreteAckley": DiscreteAckley,
    "DiscreteRosenbrock": DiscreteRosenbrock,
    "TreeFunction": TreeFunction,
    # multi-fidelity
    # "currin": CurrinExp2D,
    "SVRBench": SVRBench,
    "CCOBench": CCOBench,
    "MAXBandit": MAXBandit,
    "DatasetBenchmark": DatasetBenchmark,
}


def map_benchmark(name: str, **kwargs) -> Benchmark:
    """Map a benchmark name to a function that generates the benchmark"""
    return BENCHMARK_MAP[name](**kwargs)
