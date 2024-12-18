from bark.utils.bit_operations import sample_binary_mask


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
