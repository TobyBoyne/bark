import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from bark import types
from bark.fitting.bark_prior_sampler import sample_forest, sample_soft_forest
from bark.testing.data_test_cases import get_continuous_data_trid
from bark.tree_kernels.tree_gps import forest_predict, soft_forest_predict

jax.config.update("jax_enable_x64", True)

params = types.BARKConfig()
data = get_continuous_data_trid(N=15, dim=1)
key, subkey = jax.random.split(jax.random.key(0))

trees = sample_forest(20, data.bounds, data.feat_types, params, key)
soft_trees = sample_soft_forest(20, data.bounds, data.feat_types, params, key)

noise = jnp.array(0.01)
bark_model = types.BARKModel(trees=trees, noise_var=noise)
soft_bark_model = types.BARKModel(trees=soft_trees, noise_var=noise)

test_X = jnp.linspace(-0.5, 1.5, num=100)[:, None]

mu, var = forest_predict(bark_model, data, test_X, diag=True)

soft_mu, soft_var = soft_forest_predict(soft_bark_model, data, test_X, diag=True)

fig, ax = plt.subplots()
ax.plot(test_X.flatten(), mu.flatten())
ax.plot(test_X.flatten(), soft_mu.flatten())
plt.show()
