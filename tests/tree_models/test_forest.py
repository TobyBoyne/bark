import numpy as np

# os.environ["NUMBA_DISABLE_JIT"] = "1"
from alfalfa.forest_numba import NODE_RECORD_DTYPE, FeatureTypeEnum, forest_gram_matrix

nodes = np.array(
    [
        (0, 0, 0.5, 1, 2, 0, 1),
        (0, 0, 0.25, 3, 4, 1, 1),
        (1, 0, 1.0, 0, 0, 1, 1),
        (1, 0, 1.0, 0, 0, 2, 1),
        (1, 0, 1.0, 0, 0, 2, 1),
    ],
    dtype=NODE_RECORD_DTYPE,
).reshape(1, -1)

x1 = np.linspace(0, 1, 20).reshape(-1, 1)
x2 = np.linspace(0, 1, 20).reshape(-1, 1)

K = forest_gram_matrix(nodes, x1, x2, np.array([FeatureTypeEnum.Cont.value]))
print(K)
