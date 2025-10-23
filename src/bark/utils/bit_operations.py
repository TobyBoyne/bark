import jax
import jax.numpy as jnp
from jaxtyping import Int


@jax.jit
def next_power_of_2(x: Int[jax.Array, "..."]):
    """Returns the next integer that is a power of two.

    Specifically, calculates 2**(floor(log2(x)) + 1) if x >= 1, else
    returns 1."""
    return 1 << next_power_of_2_exponent(x)


@jax.jit
def next_power_of_2_exponent(x: Int[jax.Array, "..."]):
    """Returns the exponent of the next integer that is a power of two.

    Specifically, calculates floor(log2(x)) + 1 if x >= 1, else
    returns 0."""
    x = x.astype(jnp.uint64)
    bit_width = x.dtype.itemsize * 8
    return jnp.where(x == 0, 0, bit_width - jax.lax.clz(x))


@jax.jit
def sample_binary_mask(x: Int[jax.Array, "..."], key: jax.Array):
    """Uniformly sample each bit in the binary mask.

    This function samples a subset of the available categories, without generating
    the redundant splits:
     - split == 0
     - split == available"""
    num_choices = jnp.bitwise_count(x)

    max_sample = (1 << num_choices) - 1
    bitmask_sample = jax.random.randint(
        key, x.shape, minval=1, maxval=max_sample, dtype=x.dtype
    )

    def body(i, v):
        x, sampled_mask = v
        # check that the ith bit is 1, ie. this category is available to be sampled:
        # if so, set that bit to 0.
        is_available = (x >> i) & 1
        x &= ~(is_available << i)
        # set the ith bit to the next value in the sampled mask - 0 or 1 with equal
        # probability.
        x |= (is_available & sampled_mask & 1) << i
        sampled_mask >>= is_available

        return x, sampled_mask

    x, sampled_mask = jax.lax.fori_loop(0, 64, body, (x, bitmask_sample))
    return x
