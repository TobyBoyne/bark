import math

import pytest

from bark.utils.bit_operations import (
    next_power_of_2,
    next_power_of_2_exponent,
    sample_binary_mask,
)


def test_sample_binary_mask():
    mask = 0b100101
    for i in range(100):
        x = sample_binary_mask(mask)
        assert x != 0
        assert x != mask
        assert x & mask == x


def test_sample_binary_mask_one_category():
    mask = 0b00100
    for i in range(100):
        x = sample_binary_mask(mask)
        assert x == 0


@pytest.mark.parametrize(
    ("value", "expected", "expected_exp"),
    [
        (0, 1, 0),
        (1, 2, 1),
        (2, 4, 2),
        (3, 4, 2),
        (8, 16, 4),
        (14, 16, 4),
    ],
)
def test_next_power_of_2(value, expected, expected_exp):
    log2 = math.log2(value) if value >= 1 else -1
    calc_expected_exp = int(math.floor(log2)) + 1
    calc_expected = 2**calc_expected_exp

    assert next_power_of_2(value) == expected
    assert calc_expected == expected

    assert next_power_of_2_exponent(value) == expected_exp
    assert calc_expected_exp == expected_exp
