import torch

from alfalfa.fitting.bart.quick_inverse import low_rank_det_update, low_rank_inv_update


class TestLowRankUpdates:
    def test_low_rank_inv_update(self):
        N, B = 5, 2
        rng = torch.Generator().manual_seed(42)
        A = torch.rand((N, N), generator=rng)
        A_inv = torch.linalg.inv(A)
        U = torch.rand((N, B), generator=rng)

        woodbury_inv = low_rank_inv_update(A_inv, U)

        assert torch.isclose(woodbury_inv, torch.linalg.inv(A + U @ U.mT)).all()

    def test_low_rank_inv_update_subtract(self):
        N, B = 5, 2
        rng = torch.Generator().manual_seed(42)
        A = torch.rand((N, N), generator=rng)
        A_inv = torch.linalg.inv(A)
        U = torch.rand((N, B), generator=rng)

        woodbury_inv = low_rank_inv_update(A_inv, U, subtract=True)
        print(woodbury_inv)
        print(torch.linalg.inv(A - U @ U.mT))

        assert torch.isclose(woodbury_inv, torch.linalg.inv(A - U @ U.mT)).all()

    def test_low_rank_det_update(self):
        N, B = 5, 2
        rng = torch.Generator().manual_seed(44)
        A = torch.rand((N, N), generator=rng)
        A_inv = torch.linalg.inv(A)
        A_logdet = torch.logdet(A)
        U = torch.rand((N, B), generator=rng)

        lemma_det = low_rank_det_update(A_inv, U, A_logdet)

        assert torch.isclose(lemma_det, torch.logdet(A + U @ U.mT))

    def test_low_rank_det_update_subtract(self):
        N, B = 5, 2
        rng = torch.Generator().manual_seed(46)
        A = torch.rand((N, N), generator=rng)
        A_inv = torch.linalg.inv(A)
        A_logdet = torch.logdet(A)
        U = torch.rand((N, B), generator=rng)

        lemma_det = low_rank_det_update(A_inv, U, A_logdet, subtract=True)
        print(lemma_det)
        print(torch.logdet(A - U @ U.mT))
        assert not torch.isnan(lemma_det)
        assert torch.isclose(lemma_det, torch.logdet(A - U @ U.mT))


TestLowRankUpdates().test_low_rank_inv_update_subtract()
