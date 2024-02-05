import numpy as np


def branin(x: np.ndarray):
    """Evaluate the Branin function at the given coords.

    Args:
        x: [num_points x 2] array of coordinates.

    Returns:
        [num_points] array of function values.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return -1 * (
        a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * np.cos(x1) + s
    )


def rescaled_branin(x: np.ndarray):
    """Evaluate the Branin function at the given coords.

    Args:
        x: [num_points x 2] array of coordinates.

    Returns:
        [num_points] array of function values.
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    x1_b = 15 * x1 - 5
    x2_b = 15 * x2
    A = (x2_b - (5.1 / (4 * np.pi**2)) * x1_b**2 + (5 / np.pi) * x1_b - 6) ** 2
    B = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1_b) - 44.81
    return -(A + B) / 51.95
