import numpy as np
import lightgbm as lgbm
import matplotlib.pyplot as plt

def objective(x):
    """Objective function to be minimized."""
    return np.where(x < 0.5, 0.1*x, 1 - 0.1*x)

x = np.linspace(0, 1, 50)
y = objective(x)
model = lgbm.train(
    {"boosting_rounds": 50, "max_depth": 3, "min_data_in_leaf": 1},
    lgbm.Dataset(x.reshape(-1, 1), y),
)

print(model.predict(0.4))

x_test = np.linspace(0, 1, 1000)

plt.plot(x_test, objective(x_test))
plt.plot(x_test, model.predict(x_test.reshape(-1, 1)))
plt.scatter(x, y, marker="x")
plt.show()