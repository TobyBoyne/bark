import numpy as np
import pytest

from bark.fitting.quick_inverse import low_rank_det_update, low_rank_inv_update
from bark.forest_numba import (
    NODE_RECORD_DTYPE,
    FeatureTypeEnum,
    forest_gram_matrix,
    get_leaf_vectors,
)


def random_A_U_Ainv_Alogdet(N: int, B: int, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    U = rng.standard_normal((N, B)) * 0.1
    _, logdet = np.linalg.slogdet(A)
    return (A, U, np.linalg.inv(A), logdet)


@pytest.mark.parametrize(
    ("A", "U", "A_inv", "A_logdet"),
    [
        random_A_U_Ainv_Alogdet(5, 2, 42),
        random_A_U_Ainv_Alogdet(4, 2, 43),
        random_A_U_Ainv_Alogdet(6, 3, 44),
    ],
)
class TestLowRankUpdates:
    def test_low_rank_inv_update(self, A, U, A_inv, A_logdet):
        woodbury_inv = low_rank_inv_update(A_inv, U, subtract=False)
        assert np.isclose(woodbury_inv, np.linalg.inv(A + U @ U.T)).all()

    def test_low_rank_inv_update_subtract(self, A, U, A_inv, A_logdet):
        woodbury_inv = low_rank_inv_update(A_inv, U, subtract=True)
        assert np.isclose(woodbury_inv, np.linalg.inv(A - U @ U.T)).all()

    def test_low_rank_det_update(self, A, U, A_inv, A_logdet):
        if np.isnan(A_logdet):
            pytest.skip("Matrix A has a negative determinant.")
        lemma_det = low_rank_det_update(A_inv, U, A_logdet)
        _, logdet = np.linalg.slogdet(A + U @ U.T)
        assert not np.isnan(lemma_det).any()
        assert np.isclose(lemma_det, logdet)

    def test_low_rank_det_update_subtract(self, A, U, A_inv, A_logdet):
        if np.isnan(A_logdet):
            pytest.skip("Matrix A has a negative determinant.")
        lemma_det = low_rank_det_update(A_inv, U, A_logdet, subtract=True)
        _, logdet = np.linalg.slogdet(A - U @ U.T)
        assert not np.isnan(lemma_det)
        assert np.isclose(lemma_det, logdet)


def test_low_rank_update_with_forest():
    forest = np.zeros((2, 5), dtype=NODE_RECORD_DTYPE)
    forest[0, 0] = (1, 0, 0, 0, 0, 0, 1)
    forest[1, 0] = (0, 0, 0.5, 1, 2, 0, 1)
    forest[1, 1] = (0, 0, 0.25, 3, 4, 1, 1)
    forest[1, 2] = (1, 0, 0, 0, 0, 1, 1)
    forest[1, 3] = (1, 0, 0, 0, 0, 2, 1)
    forest[1, 4] = (1, 0, 0, 0, 0, 2, 1)

    new_nodes = forest[0].copy()
    new_nodes[0] = (0, 0, 0.75, 1, 2, 0, 1)
    new_nodes[1] = (1, 0, 0, 0, 0, 1, 1)
    new_nodes[2] = (1, 0, 0, 0, 0, 1, 1)

    train_x = np.linspace(0, 1, 20).reshape(-1, 1)
    feat_types = np.array([FeatureTypeEnum.Cont.value])

    scale = 0.5
    noise = 0.1
    K_XX = scale * forest_gram_matrix(forest, train_x, train_x, feat_types)
    K_XX_s = K_XX + noise * np.eye(K_XX.shape[0])
    K_inv = np.linalg.inv(K_XX_s)
    _, K_logdet = np.linalg.slogdet(K_XX_s)

    s_sqrtm = np.sqrt(scale / forest.shape[0])

    cur_leaf_vectors = s_sqrtm * get_leaf_vectors(forest[0], train_x, feat_types)
    new_leaf_vectors = s_sqrtm * get_leaf_vectors(new_nodes, train_x, feat_types)

    new_K_inv, new_K_logdet = (
        low_rank_inv_update(K_inv, cur_leaf_vectors, subtract=True),
        low_rank_det_update(K_inv, cur_leaf_vectors, K_logdet, subtract=True),
    )

    new_K_inv, new_K_logdet = (
        low_rank_inv_update(new_K_inv, new_leaf_vectors),
        low_rank_det_update(new_K_inv, new_leaf_vectors, new_K_logdet),
    )

    forest[0] = new_nodes
    K_XX = scale * forest_gram_matrix(forest, train_x, train_x, feat_types)
    K_XX_s = K_XX + noise * np.eye(K_XX.shape[0])
    K_inv_exact = np.linalg.inv(K_XX_s)
    _, K_logdet_exact = np.linalg.slogdet(K_XX_s)

    assert np.isclose(K_logdet_exact, new_K_logdet)
    assert np.isclose(K_inv_exact, new_K_inv).all()
