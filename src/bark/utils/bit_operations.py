import numpy as np
from numba import njit


@njit
def next_power_of_2(x):
    """Returns the next integer that is a power of two.

    Specifically, calculates 2**(floor(log2(x)) + 1) if x >= 1, else
    returns 1."""
    p = 1
    while x >= p:
        p <<= 1
    return p


@njit
def next_power_of_2_exponent(x):
    """Returns the exponent of the next integer that is a power of two.

    Specifically, calculates floor(log2(x)) + 1 if x >= 1, else
    returns 0."""
    p = 1
    e = 0
    while x >= p:
        p <<= 1
        e += 1
    return e


@njit
def bit_count(x: int):
    """Number of bits set to 1 in the binary representation of x."""

    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


@njit
def sample_binary_mask(x: int):
    """Uniformly sample each bit in the binary mask.

    This function samples a subset of the available categories, without generating
    the redundant splits:
     - split == 0
     - split == available"""
    num_choices = bit_count(x)

    # there must be at least two categories to choose from
    if num_choices < 2:
        return 0

    max_sample = (1 << num_choices) - 1
    bitmask_sample = np.random.randint(1, max_sample)
    threshold = 0

    for i in range(next_power_of_2_exponent(x)):
        if x & (1 << i):
            bitmask_selected = bitmask_sample & 1
            threshold |= bitmask_selected << i
            bitmask_sample >>= 1

    return threshold
