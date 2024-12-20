import matplotlib.pyplot as plt
import numpy as np
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput

from bark.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from bark.forest import create_empty_forest
from bark.tree_kernels.tree_gps import forest_predict

bark_params = BARKTrainParams(
    warmup_steps=500,
    n_steps=100,
    thinning=50,
    num_chains=4,
)

forest = create_empty_forest(m=10)
forest = np.tile(forest, (bark_params.num_chains, 1, 1))
noise = np.tile(0.1, (bark_params.num_chains,))
scale = np.tile(1.0, (bark_params.num_chains,))

train_x = np.linspace(0.5, 1, 20)[:, None]
train_y = train_x - 0.5

domain = Domain.from_lists(
    inputs=[ContinuousInput(key="x", bounds=(0, 1))],
    outputs=[ContinuousOutput(key="y")],
)

samples = run_bark_sampler(
    model=(forest, noise, scale),
    data=(train_x, train_y),
    domain=domain,
    params=bark_params,
)

test_x = np.linspace(0, 1, 100)[:, None]
mu_samples, var_samples = forest_predict(samples, (train_x, train_y), test_x, domain)
mu, var = mu_samples.mean(axis=0), var_samples.mean(axis=0)

fig, ax = plt.subplots()
ax.plot(test_x.flatten(), mu)
ax.fill_between(
    test_x.flatten(), mu - 2 * np.sqrt(var), mu + 2 * np.sqrt(var), alpha=0.2
)
ax.scatter(train_x.flatten(), train_y.flatten(), c="black", marker="x")

fig, ax = plt.subplots()
ax.plot(test_x.flatten(), var)
plt.show()
