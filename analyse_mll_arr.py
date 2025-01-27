import matplotlib.pyplot as plt
import numpy as np

mll_arr = np.load("mll_arr.npy")

plt.plot(mll_arr)
plt.show()
