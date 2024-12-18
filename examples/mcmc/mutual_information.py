import numpy as np


def mutual_information_mc(forest_1, forest_2):
    x = np.random.uniform(-5, 5, 1000)
    # leaves_1 = get_leaf_vectors(forest_1)
    return x


with open(r"C:\Users\tobyb\phd\bark\forest.npy", "rb") as f:
    forest = np.load(f)

print(forest.shape)
x = mutual_information_mc(forest[0, 0], forest[0, 1])
print(x.shape)
