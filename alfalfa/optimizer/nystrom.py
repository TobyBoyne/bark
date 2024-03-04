import torch

from alfalfa.tree_kernels import AlfalfaGP


def construct_nystrom_features(model: AlfalfaGP, x_samples: torch.Tensor):
    # K_hat is (m x m)
    K_hat = model.covar_module(x_samples)
    K_hat_inv = torch.linalg.pinv(K_hat.to_dense())

    def nystrom_vector_func(x: torch.Tensor):
        # x is (N x D)
        # K_b is (N x m)
        K_b = model.covar_module(x, x_samples)
        K_r = K_b @ K_hat_inv @ K_b.mT
        # is K_r hermitian?
        # pytorch returns in ascending order, we want descending
        L, Q = torch.linalg.eigh(K_r)
        L, Q = torch.flip(L, dims=[-1]), torch.flip(Q, dims=[-1])

        # calculate the rank of k from the eigenvalues
        # all zero values of L are equal to 1e-10
        L = torch.clamp(L, 1e-10, torch.inf)
        r = torch.argmin(L)

        D = torch.diag(L[:r] ** (-1 / 2))
        Q = Q[:, :r]

        # D is (r x r)
        # Q.T is (r x m)
        # K_b.T is (m x N)
        # => z is (r x N)

        z = D @ Q.mT @ K_b.mT
        return z

    return nystrom_vector_func


def nystrom_samples(nystrom_vector: torch.Tensor, num_samples: int):
    weights = torch.randn((nystrom_vector.shape[0], num_samples))
    return weights.T @ nystrom_vector
