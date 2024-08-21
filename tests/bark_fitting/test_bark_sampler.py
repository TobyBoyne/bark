import numpy as np

from alfalfa.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from alfalfa.forest_numba import NODE_RECORD_DTYPE

nodes = np.zeros((50, 100), dtype=NODE_RECORD_DTYPE)

params = BARKTrainParams()
samples = run_bark_sampler()
