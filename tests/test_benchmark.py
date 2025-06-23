import numpy as np
import pytest
from beartype.typing import Type
from bofire.benchmarks.benchmark import Benchmark

from bofire_mixed.benchmarks import BENCHMARK_MAP

BENCHMARKS = list(BENCHMARK_MAP.values())


@pytest.mark.parametrize("benchmark_cls", BENCHMARKS)
def test_benchmark_optima_correctness(benchmark_cls: Type[Benchmark]):
    benchmark = benchmark_cls()
    try:
        opt = benchmark.get_optima()
    except NotImplementedError:
        pytest.skip()
    X, y = (
        opt[benchmark.domain.inputs.get_keys()],
        opt[benchmark.domain.outputs.get_keys()],
    )
    assert benchmark.domain.constraints.is_fulfilled(X, tol=1e-2).all()
    assert np.isclose(y, benchmark.f(X), rtol=1e-2).all()
