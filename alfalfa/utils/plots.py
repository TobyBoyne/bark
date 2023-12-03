import gpytorch as gpy
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_gp_1d(model: gpy.kernels.Kernel, likelihood: gpy.likelihoods.Likelihood, 
            train_x, train_y):
    with torch.no_grad():

        # predictions
        test_x = torch.linspace(0, 1, 500)
        observed_pred = likelihood(model(test_x))

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

    return ax


def plot_gp_2d(model: gpy.kernels.Kernel, likelihood: gpy.likelihoods.Likelihood, 
            train_x, train_y, test_X, target):
    """Plot a GP with two input dimensions."""
    fig, axs = plt.subplots(ncols=3, figsize=(8, 3))    

    with torch.no_grad():
        # predictions
        test_X1, test_X2 = test_X
        test_x = torch.stack([test_X1.reshape(-1), test_X2.reshape(-1)], dim=1)
        observed_pred = likelihood(model(test_x))

        k = 1.0
        ys = (observed_pred.mean, observed_pred.variance, observed_pred.mean + k*observed_pred.variance)
        for ax, y in zip(axs, ys):
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