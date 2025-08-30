import timeit
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float

from bark.fitting.quick_inverse import LowRankInverter, mll

jax.config.update("jax_enable_x64", True)


def generate_random_problem(
    N: int, B: int, seed: int
) -> tuple[Float[jax.Array, "{N} {N}"], LowRankInverter]:
    key = jax.random.key(seed)
    keys = jax.random.split(key, num=3)
    A_rt = jax.random.normal(keys[0], (N, N), dtype=jnp.float64)
    U = jax.random.normal(keys[1], (N, B), dtype=jnp.float64) * 0.1
    y = jax.random.normal(keys[2], (N, 1), dtype=jnp.float64)
    A = A_rt @ A_rt.T
    A_inv = jnp.linalg.inv(A)
    _, A_logdet = np.linalg.slogdet(A)
    A_mll = mll(A_inv, A_logdet, y)
    return A, LowRankInverter(
        K_inv=A_inv, K_logdet=A_logdet, mll=A_mll, U=U, subtract=False, y=y
    )


@pytest.mark.parametrize(
    ("A", "lr_inverter"),
    [
        generate_random_problem(5, 2, seed=0),
        generate_random_problem(4, 2, seed=1),
        generate_random_problem(50, 3, seed=2),
    ],
)
class TestLowRankUpdates:
    def test_low_rank_update_correctness(
        self, A: jax.Array, lr_inverter: LowRankInverter
    ):
        U = lr_inverter.U
        A_update = A + U @ U.T
        lr_update = lr_inverter.low_rank_update()

        A_update_inv = jnp.linalg.inv(A_update)
        _, A_update_logdet = jnp.linalg.slogdet(A_update)
        A_update_mll = mll(A_update_inv, A_update_logdet, lr_inverter.y)

        assert jnp.isclose(
            lr_update.K_inv @ A_update, jnp.eye(A.shape[0]), atol=1e-5
        ).all()
        assert jnp.isclose(A_update_logdet, lr_update.K_logdet, rtol=1e-5)
        assert jnp.isclose(A_update_mll, lr_update.mll, rtol=1e-5)

        lr_inverter_sub = replace(lr_inverter, subtract=True)
        A_update = A - U @ U.T
        lr_update = lr_inverter_sub.low_rank_update()

        A_update_inv = jnp.linalg.inv(A_update)
        _, A_update_logdet = jnp.linalg.slogdet(A_update)
        A_update_mll = mll(A_update_inv, A_update_logdet, lr_inverter.y)

        assert jnp.isclose(
            lr_update.K_inv @ A_update, jnp.eye(A.shape[0]), atol=1e-5
        ).all()
        assert jnp.isclose(
            jnp.linalg.slogdet(A_update)[1], lr_update.K_logdet, rtol=1e-5
        )
        assert jnp.isclose(A_update_mll, lr_update.mll, rtol=1e-5)

    def test_low_rank_update_reversible(
        self, A: jax.Array, lr_inverter: LowRankInverter
    ):
        # Applying the low rank update once returns the lr_inverter that recovers
        # the initial matrices. Applying again will therefore produce the same matrices.
        lr_update_twice = lr_inverter.low_rank_update().low_rank_update()
        assert jnp.isclose(
            lr_update_twice.K_inv @ A, jnp.eye(A.shape[0]), atol=1e-5
        ).all()
        assert jnp.isclose(lr_inverter.K_logdet, lr_update_twice.K_logdet, rtol=1e-5)
        assert jnp.isclose(lr_inverter.mll, lr_update_twice.mll, rtol=1e-5)


@pytest.mark.parametrize(
    ("A", "lr_inverter"),
    [
        generate_random_problem(100, 3, seed=2),
    ],
)
def test_low_rank_update_speed(A: jax.Array, lr_inverter: LowRankInverter):
    # The speed difference is most significant for large values of N
    def slow_update():
        U = lr_inverter.U
        A_update = A + U @ U.T
        A_update_inv = jnp.linalg.inv(A_update)
        _, A_update_logdet = jnp.linalg.slogdet(A_update)
        _ = mll(A_update_inv, A_update_logdet, lr_inverter.y)

    _ = lr_inverter.low_rank_update()
    times_lr = timeit.repeat(lr_inverter.low_rank_update, number=100)
    times_default = timeit.repeat(slow_update, number=100)

    # We find a speed improvement of ~100x
    assert sum(times_lr) < sum(times_default)
