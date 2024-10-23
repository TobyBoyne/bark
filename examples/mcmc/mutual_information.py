import numpy as np

from bark.benchmarks import StyblinskiTang
from bark.bofire_utils.domain import get_feature_bounds, get_feature_types_array
from bark.fitting.diagnostics import mutual_information_forest_pair

with open(r"C:\Users\tobyb\phd\bark\forest.npy", "rb") as f:
    forest = np.load(f)

domain = StyblinskiTang().domain
bounds = np.array(
    [get_feature_bounds(feat, encoding="bitmask") for feat in domain.inputs.get()]
)
feat_types = get_feature_types_array(domain)

forest_samples = forest.reshape(-1, *forest.shape[-2:])

mutual_information_matrix = np.zeros((forest_samples.shape[0], forest_samples.shape[0]))
# for i in range(forest_samples.shape[0]):
#     for j in range(i, forest_samples.shape[0]):
#         mutual_information_matrix[i, j] = mutual_information(forest_samples[i, 0], forest_samples[j, 0], bounds, feat_types)
mi_01 = mutual_information_forest_pair(
    forest_samples[0], forest_samples[1], bounds, feat_types
)
mi_00 = mutual_information_forest_pair(
    forest_samples[0], forest_samples[0], bounds, feat_types
)
print(mi_01, mi_00)

# plt.imshow(mutual_information_matrix)
# plt.show()
