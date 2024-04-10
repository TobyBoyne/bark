import warnings

import gpytorch as gpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from beartype.typing import Callable, Optional


def plot_gp_nd(model: gpy.models.ExactGP, test_x, target: Callable, ax=None, D=None):
    # infer dimension of input from train x
    if D is None:
        D = model.train_inputs[0].shape[-1]

    if D == 1:
        return plot_gp_1d(model, test_x, target, ax)
    elif D == 2:
        return plot_gp_2d(model, test_x, target)
    else:
        warnings.warn("You can only plot GPs with 1 or 2 input dimensions.")


def plot_gp_1d(
    model: gpy.models.ExactGP,
    test_x,
    target: Optional[Callable[[np.ndarray], np.ndarray]],
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()
    with torch.no_grad():
        likelihood = model.likelihood
        train_x = model.train_inputs[0]
        train_y = model.train_targets
        # predictions
        observed_pred = likelihood(model(test_x))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), "k*", label="train points", zorder=10)
        # Plot predictive means as blue line
        ax.plot(
            test_x.flatten().numpy(),
            observed_pred.mean.numpy(),
            "b",
            label="predictive mean",
        )
        # Shade between the lower and upper confidence bounds
        ax.fill_between(
            test_x.flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5
        )
        if target is not None:
            ax.plot(test_x.flatten().numpy(), target(test_x), label="target function")
        ax.legend()

        # ax = axs[1]
        # # plot *just* the variance
        # ax.plot(test_x.flatten().numpy(), (upper - observed_pred.mean).numpy())

    return ax


def plot_gp_2d(model: gpy.models.ExactGP, test_X, target: Callable):
    """Plot a GP with two input dimensions."""
    fig, axs = plt.subplots(ncols=3, figsize=(8, 3))
    axs: list[plt.Axes]
    likelihood = model.likelihood
    train_x = model.train_inputs[0]
    with torch.no_grad():
        # predictions
        test_X1, test_X2 = test_X
        test_x = torch.stack([test_X1.reshape(-1), test_X2.reshape(-1)], dim=1)
        observed_pred = likelihood(model(test_x))

        k = 1.0
        ys = (
            observed_pred.mean,
            observed_pred.variance,
            observed_pred.mean + k * observed_pred.variance,
        )
        for ax, y in zip(axs, ys):
            # Plot training data as black stars
            ax.plot(train_x.numpy()[:, 0], train_x.numpy()[:, 1], "k*")
            # Filled contours of GP predictions
            ax.contourf(test_X1, test_X2, y.numpy().reshape(test_X1.shape))
            # Contour of target function
            ax.contour(
                test_X1,
                test_X2,
                target(
                    torch.stack((test_X1.flatten(), test_X2.flatten()), dim=1)
                ).reshape(test_X1.shape),
            )

        axs[0].set_title("Mean")
        axs[1].set_title("Variance")
        axs[2].set_title(f"UCB ($\kappa={k}$)")
    return fig, axs


def plot_covar_matrix(model: gpy.models.ExactGP, test_x, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    with torch.no_grad():
        cov = model.covar_module(test_x).evaluate().numpy()
        im = ax.imshow(cov, interpolation="nearest")

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, im
