import logging
import sys

import bofire.strategies.api as strategies
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# from bark.benchmarks import
from bofire.benchmarks.single import Hartmann
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy
from matplotlib.colors import Normalize

from bark.fitting.bark_sampler import BARKTrainParams, run_bark_sampler
from bark.forest import batched_forest_gram_matrix, create_empty_forest
from bofire_mixed.domain import get_feature_types_array

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    benchmark = Hartmann()
    domain = benchmark.domain

    # sample initial points
    sampler = strategies.map(RandomStrategy(domain=domain, seed=0))
    train_x = sampler.ask(70)
    train_y = benchmark.f(train_x)["y"]  # .drop("valid_y", axis="columns")
    cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)
    transform_specs = {k: CategoricalEncodingEnum.ORDINAL for k in cat}

    feat_types = get_feature_types_array(domain)

    bark_params = BARKTrainParams(warmup_steps=1, num_samples=100, steps_per_sample=1)

    forest = create_empty_forest(m=50)
    forest = np.tile(forest, (bark_params.num_chains, 1, 1))
    noise = np.tile(0.1, (bark_params.num_chains,))
    scale = np.tile(1.0, (bark_params.num_chains,))

    train_x_transformed = domain.inputs.transform(train_x, transform_specs)
    train_x_transformed = train_x_transformed.to_numpy()
    train_y_transformed = ((train_y - train_y.mean()) / train_y.std()).to_numpy()[
        :, None
    ]
    data_numpy = (train_x_transformed, train_y_transformed)

    samples = run_bark_sampler(
        model=(forest, noise, scale),
        data=data_numpy,
        domain=domain,
        params=bark_params,
    )

    # update starting values
    forest = samples[0][:, -1, :, :]
    noise = samples[1][:, -1]
    scale = samples[2][:, -1]

    gram_matrix = batched_forest_gram_matrix(
        forest, train_x_transformed, train_x_transformed, feat_types
    )
    # gram matrix is (1, 1, N, N)
    print(noise, scale)

    resolution = 20
    scale_grid, noise_grid = np.meshgrid(
        np.linspace(0.3, 5.0, resolution), np.linspace(0.01, 1.0, resolution)
    )
    scale_grid = scale_grid[..., None, None]
    noise_grid = noise_grid[..., None, None]
    K_XX = scale_grid * gram_matrix[None, ...]
    K_XX_s = K_XX + noise_grid * np.eye(train_x_transformed.shape[0])[None, None, ...]
    K_inv = np.linalg.inv(K_XX_s)
    _, K_logdet = np.linalg.slogdet(K_XX_s)

    data_fit = -0.5 * (train_y_transformed.T @ K_inv @ train_y_transformed)
    complexity = -0.5 * K_logdet[..., None, None]
    mll = data_fit + complexity
    cmap = mpl.colormaps.get_cmap("viridis")

    all_values = np.concatenate(
        [data_fit.flatten(), complexity.flatten(), mll.flatten()]
    )
    min_cmap = np.percentile(all_values, 1)
    max_cmap = np.percentile(all_values, 99)

    normalizer = Normalize(min_cmap, max_cmap)
    im = cm.ScalarMappable(norm=normalizer)

    fig, axs = plt.subplots(ncols=3)
    for i, (ax, z) in enumerate(zip(axs, (data_fit, complexity, mll))):
        ax: plt.Axes
        x, y = scale_grid[..., 0, 0], noise_grid[..., 0, 0]
        ax.contourf(x, y, z[..., 0, 0], cmap=cmap, norm=normalizer, levels=20)
        ax.set_title(["Data fit", "Complexity", "Marginal log likelihood"][i])

    axs[0].set_xlabel("Scale")
    axs[0].set_ylabel("Noise")
    fig.colorbar(im, ax=axs.tolist())

    plt.show()

    print("...")


if __name__ == "__main__":
    main()
