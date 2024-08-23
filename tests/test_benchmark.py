import numpy as np
import pytest
from beartype.typing import Type
from bofire.benchmarks.benchmark import Benchmark

from bark.benchmarks import BENCHMARK_MAP
from bark.bofire_utils.sampling import sample_projected

BENCHMARKS = list(BENCHMARK_MAP.values())


@pytest.mark.parametrize("benchmark_cls", BENCHMARKS)
def test_sampling_benchmark(benchmark_cls: Type[Benchmark]):
    benchmark = benchmark_cls()
    samples = sample_projected(benchmark.domain, n=10, seed=42)
    evals = benchmark.f(samples)  # noqa: F841

    assert benchmark.domain.constraints.is_fulfilled(samples, tol=1e-3).all()


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
