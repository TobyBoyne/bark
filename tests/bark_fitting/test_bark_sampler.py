import gpytorch as gpy
import numpy as np
import torch
from bofire.benchmarks.api import Hartmann

# os.environ["NUMBA_DISABLE_JIT"] = "1"
from alfalfa.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from alfalfa.forest_numba import NODE_RECORD_DTYPE
from alfalfa.tree_kernels.tree_gps import AlfalfaGPNumba

nodes = np.zeros((50, 100), dtype=NODE_RECORD_DTYPE)
nodes[:, 0]["active"] = 1
nodes[:, 0]["is_leaf"] = 1


benchmark = Hartmann()
train_x = benchmark.domain.inputs.sample(10)
train_y = benchmark.f(train_x)[["y"]]

likelihood = gpy.likelihoods.GaussianLikelihood()
gp = AlfalfaGPNumba(
    torch.from_numpy(train_x.to_numpy()),
    torch.from_numpy(train_y.to_numpy()),
    likelihood,
    nodes,
)

params = BARKTrainParams(warmup_steps=2, n_steps=2)
print("sampling")
samples = run_bark_sampler(gp, benchmark.domain, params)
print(samples)
