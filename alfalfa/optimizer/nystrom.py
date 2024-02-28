import torch

from alfalfa.tree_kernels import AlfalfaGP


def construct_nystrom_features(model: AlfalfaGP, x_samples: torch.Tensor):
    K_hat = model.covar_module(x_samples)
    K_hat_inv = torch.linalg.pinv(K_hat)

    def nystrom_vector_func(x: torch.Tensor):
        # x is (N x D)
        K_b = model.covar_module(x, x_samples)
        K_r = K_b @ K_hat_inv @ K_b.mT
        # is K_r hermitian?
        # pytorch returns in ascending order, we want descending
        L, Q = torch.linalg.eigh(K_r)
        L, Q = L[..., ::-1], Q[..., ::-1]
        D = torch.diag(L ** (-1 / 2))

        z = D @ Q.mT @ K_b.mT
        return z

    return nystrom_vector_func


def nystrom_samples(nystrom_vector: torch.Tensor, num_samples: int):
    weights = torch.randn((nystrom_vector.shape[0], num_samples))
    return weights.T @ nystrom_vector
