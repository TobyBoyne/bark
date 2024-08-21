import pytest
import torch

from alfalfa.fitting.bart.quick_inverse import low_rank_det_update, low_rank_inv_update


def random_A_U_Ainv_Alogdet(N: int, B: int, seed=42):
    rng = torch.Generator().manual_seed(seed)
    A = torch.rand((N, N), generator=rng)
    U = torch.rand((N, B), generator=rng) * 0.1

    return (A, U, torch.linalg.inv(A), torch.logdet(A))


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
        woodbury_inv = low_rank_inv_update(A_inv, U)
        assert torch.isclose(woodbury_inv, torch.linalg.inv(A + U @ U.mT)).all()

    def test_low_rank_inv_update_subtract(self, A, U, A_inv, A_logdet):
        woodbury_inv = low_rank_inv_update(A_inv, U, subtract=True)
        assert torch.isclose(woodbury_inv, torch.linalg.inv(A - U @ U.mT)).all()

    def test_low_rank_det_update(self, A, U, A_inv, A_logdet):
        if torch.isnan(A_logdet):
            pytest.skip("Matrix A has a negative determinant.")
        lemma_det = low_rank_det_update(A_inv, U, A_logdet)
        assert not torch.isnan(lemma_det)
        assert torch.isclose(lemma_det, torch.logdet(A + U @ U.mT))

    def test_low_rank_det_update_subtract(self, A, U, A_inv, A_logdet):
        if torch.isnan(A_logdet):
            pytest.skip("Matrix A has a negative determinant.")
        lemma_det = low_rank_det_update(A_inv, U, A_logdet, subtract=True)
        assert not torch.isnan(lemma_det)
        assert torch.isclose(lemma_det, torch.logdet(A - U @ U.mT))
