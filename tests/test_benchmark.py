import numpy as np

from alfalfa.benchmarks import Branin


def test_grid_sample():
    benchmark = Branin()
    xs, _ = benchmark.grid_sample(np.array([2, 3]))
    true_xs = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )
    assert np.array_equal(xs, true_xs)
