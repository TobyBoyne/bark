"""This is a suggestion for speeding up matrix inverses (for MLL).
However, it doesn't seem to be much faster..."""

from timeit import timeit

import torch


def inverse_sum(A, B):
    return torch.linalg.inv(A + B)


def fast_inverse_sum_rank1(A_inv, B):
    g = torch.trace(B @ A_inv)
    return A_inv - 1 / (1 + g) * A_inv @ B @ A_inv


def fast_inverse_sum(A_inv, Bs):
    C_inv = A_inv
    for B in Bs:
        C_inv = fast_inverse_sum_rank1(C_inv, B)
    return C_inv


def fast_inverse_sum_rank1_vec(A_inv, b):
    g = b.T @ (A_inv @ b)
    return A_inv - 1 / (1 + g) * (A_inv @ b) @ (b.T @ A_inv)


def fast_inverse_sum(A_inv, Bs):
    C_inv = A_inv
    for B in Bs:
        C_inv = fast_inverse_sum_rank1(C_inv, B)
    return C_inv


N = 20
torch.manual_seed(42)
L = torch.rand((N, N))
A = L @ L.T
A_inv = torch.linalg.inv(A)


# K = model.covar_module(x).evaluate()
# N = K.shape[0]
# sigma = model.likelihood.noise * torch.eye(N)
# K_inv = torch.linalg.inv(K + sigma)
# loss_manual = - y @ K_inv @ y - torch.logdet(K + sigma)


# # Rank 1
# B = torch.zeros_like(A)
# t = 2
# B[:t, :t] = 1
# print(timeit(lambda: inverse_sum(A, B), number=10_000))
# print(timeit(lambda: fast_inverse_sum_rank1(A_inv, B), number=10_000))

# t = N//3
# b = (torch.arange(N) < t).float().reshape((-1, 1))
# Bs = torch.stack((b @ b.T, (1-b) @ (1-b).T))
# B = torch.sum(Bs, dim=0)
# print(timeit(lambda: inverse_sum(A, B), number=10_000))
# print(timeit(lambda: fast_inverse_sum(A_inv, Bs), number=10_000))

## vectorized

# Rank 1
B = torch.zeros_like(A)
t = 2
b = (torch.arange(N) < t).float().reshape((-1, 1))
B = b @ b.T
print(timeit(lambda: inverse_sum(A, B), number=10_000))
print(timeit(lambda: fast_inverse_sum_rank1_vec(A_inv, b), number=10_000))

t = N // 3
b = (torch.arange(N) < t).float().reshape((-1, 1))
Bs = torch.stack((b @ b.T, (1 - b) @ (1 - b).T))
B = torch.sum(Bs, dim=0)
print(timeit(lambda: inverse_sum(A, B), number=10_000))
print(timeit(lambda: fast_inverse_sum(A_inv, Bs), number=10_000))
