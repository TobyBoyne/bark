from time import perf_counter

import gpytorch as gpy
import numpy as np
import torch
from bofire.benchmarks.api import Hartmann

# os.environ["NUMBA_DISABLE_JIT"] = "1"
from bark.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from bark.forest_numba import NODE_RECORD_DTYPE
from bark.tree_kernels.tree_gps import BARKGPNumba

nodes = np.zeros((50, 100), dtype=NODE_RECORD_DTYPE)
nodes[:, 0]["active"] = 1
nodes[:, 0]["is_leaf"] = 1


benchmark = Hartmann()
train_x = benchmark.domain.inputs.sample(20)
train_y = benchmark.f(train_x)[["y"]]

likelihood = gpy.likelihoods.GaussianLikelihood()
gp = BARKGPNumba(
    torch.from_numpy(train_x.to_numpy()),
    torch.from_numpy(train_y.to_numpy()),
    likelihood,
    nodes,
)

params = BARKTrainParams(warmup_steps=2, n_steps=2, thinning=1)
print("sampling")
samples = run_bark_sampler(gp, benchmark.domain, params)
print(samples)

params.n_steps = 50
params.thinning = 5
tic = perf_counter()
samples = run_bark_sampler(gp, benchmark.domain, params)
print(samples)
print(perf_counter() - tic)
