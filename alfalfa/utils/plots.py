import gpytorch as gpy
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from .logger import Logger

def plot_gp_1d(model: gpy.models.ExactGP, test_x, target: Callable):
    with torch.no_grad():
        likelihood = model.likelihood
        train_x = model.train_inputs[0]
        train_y = model.train_targets
        # predictions
        observed_pred = likelihood(model(test_x))

        # Initialize plot
        fig, axs = plt.subplots(nrows=2)
        ax = axs[0]
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label="test points")
        # Plot predictive means as blue line
        ax.plot(test_x.flatten().numpy(), observed_pred.mean.numpy(), 'b', label="predictive mean")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        ax.plot(test_x.flatten().numpy(), target(test_x), label="target function")
        ax.legend()

        ax = axs[1]
        # plot *just* the variance
        ax.plot(test_x.flatten().numpy(), (upper - observed_pred.mean).numpy())

    return fig, ax


def plot_gp_2d(model: gpy.models.ExactGP, test_X, target: Callable):
    """Plot a GP with two input dimensions."""
    fig, axs = plt.subplots(ncols=3, figsize=(8, 3))    

    likelihood = model.likelihood
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    with torch.no_grad():
        # predictions
        test_X1, test_X2 = test_X
        test_x = torch.stack([test_X1.reshape(-1), test_X2.reshape(-1)], dim=1)
        observed_pred = likelihood(model(test_x))

        k = 1.0
        ys = (observed_pred.mean, observed_pred.variance, observed_pred.mean + k*observed_pred.variance)
        for ax, y in zip(axs, ys):
            ax: plt.Axes
            # Plot training data as black stars
            ax.plot(train_x.numpy()[:, 0], train_x.numpy()[:, 1], 'k*')
            # Filled contours of GP predictions
            ax.contourf(test_X1, test_X2, y.numpy().reshape(test_X1.shape))
            # Contour of target function
            ax.contour(test_X1, test_X2, 
                target(torch.stack((test_X1.flatten(), test_X2.flatten()), dim=1)).reshape(test_X1.shape)
            )

        axs[0].set_title("Mean")
        axs[1].set_title("Variance")
        axs[2].set_title(f"UCB ($\kappa={k}$)")
    return fig, axs

def plot_covar_matrix(model: gpy.models.ExactGP, test_x):
    with torch.no_grad():
        fig, ax = plt.subplots()

        cov = model.covar_module(test_x).evaluate().numpy()
        ax.imshow(cov, interpolation="nearest")
    return fig, ax


def plot_loss_logs(logger: Logger, loss_key: str, step_key: str, test_loss_key: str):
    loss = np.array(logger[loss_key])
    steps = np.array(logger[step_key])
    fig, ax = plt.subplots()
    for step_type in np.unique(steps):
        filtered_loss = np.where(steps == step_type, loss, np.nan)
        ax.plot(filtered_loss, label=step_type)

    test_loss_xs = np.concatenate((
        np.array([0]),
        (1 + np.argwhere(steps[1:] != steps[:-1]).flatten()),
        np.array([steps.shape[0]]),
    ))
    ax.plot(test_loss_xs, logger[test_loss_key], label="test loss")
    ax.legend()

    return fig, ax