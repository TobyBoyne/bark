import matplotlib.pyplot as plt
import numpy as np

from bark.benchmarks import StyblinskiTang
from bark.bofire_utils.domain import get_feature_bounds, get_feature_types_array
from bark.fitting.diagnostics import mutual_information_forest_pair

with open(r"C:\Users\tobyb\phd\bark\forest.npy", "rb") as f:
    forest_samples = np.load(f)

domain = StyblinskiTang().domain
bounds = np.array(
    [get_feature_bounds(feat, encoding="bitmask") for feat in domain.inputs.get()]
)
feat_types = get_feature_types_array(domain)

mutual_information_matrix = np.zeros((forest_samples.shape[0], forest_samples.shape[1]))
for i in range(forest_samples.shape[0]):
    for j in range(forest_samples.shape[1]):
        mutual_information_matrix[i, j] = mutual_information_forest_pair(
            forest_samples[0, 0], forest_samples[i, j], bounds, feat_types
        )
print(mutual_information_matrix[0, 0])
plt.imshow(mutual_information_matrix)
plt.show()
