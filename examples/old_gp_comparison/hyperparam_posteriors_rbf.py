import logging
import sys

import bofire.strategies.api as strategies
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.strategies.api import RandomStrategy

# from bark.benchmarks import
from gpytorch.kernels import RBFKernel
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    benchmark = Himmelblau()
    domain = benchmark.domain

    # sample initial points
    sampler = strategies.map(RandomStrategy(domain=domain, seed=0))
    train_x = sampler.ask(30)
    train_y = benchmark.f(train_x)["y"]  # .drop("valid_y", axis="columns")
    train_y = train_y
    cat = benchmark.domain.inputs.get_keys(includes=CategoricalInput)
    transform_specs = {k: CategoricalEncodingEnum.ORDINAL for k in cat}

    train_x_transformed = domain.inputs.transform(train_x, transform_specs)
    train_x_transformed = train_x_transformed.to_numpy()
    train_y_transformed = ((train_y - train_y.mean()) / train_y.std()).to_numpy()[
        :, None
    ]
    train_y_transformed = train_y_transformed + np.random.normal(0, 0.5, train_y.shape)

    train_x_torch = torch.tensor(train_x_transformed, dtype=torch.float64)
    gram_matrix = RBFKernel()(train_x_torch, train_x_torch)
    print(RBFKernel().lengthscale.item())
    gram_matrix = gram_matrix.numpy()[None, ...]
    # gram matrix is (1, N, N)

    resolution = 20
    scale_grid, noise_grid = np.meshgrid(
        np.linspace(0.3, 3.0, resolution), np.linspace(0.01, 1.0, resolution)
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
        ax.contourf(x, y, z[..., 0, 0], levels=20, cmap=cmap, norm=normalizer)
        ax.set_title(["Data fit", "Complexity", "Marginal log likelihood"][i])
    # cf = axs[0].contourf(scale_grid[:, :, 0, 0], noise_grid[:, :, 0, 0], data_fit[:, :, 0, 0])
    # axs[1].contourf(scale_grid[:, :, 0, 0], noise_grid[:, :, 0, 0], complexity[:, :, 0, 0])
    # axs[2].contourf(scale_grid[:, :, 0, 0], noise_grid[:, :, 0, 0], mll[:, :, 0, 0])

    axs[0].set_xlabel("Scale")
    axs[0].set_ylabel("Noise")
    fig.colorbar(im, ax=axs.tolist())

    plt.show()

    print("...")


if __name__ == "__main__":
    main()
