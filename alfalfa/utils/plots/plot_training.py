import matplotlib.pyplot as plt
import numpy as np

from ..logger import Logger


def plot_loss_logs(logger: Logger, loss_key: str, step_key: str, test_loss_key: str):
    loss = np.array(logger[loss_key])
    steps = np.array(logger[step_key])
    fig, ax = plt.subplots()
    for step_type in np.unique(steps):
        filtered_loss = np.where(steps == step_type, loss, np.nan)
        ax.plot(filtered_loss, label=step_type)

    test_loss_xs = np.concatenate(
        (
            np.array([0]),
            (1 + np.argwhere(steps[1:] != steps[:-1]).flatten()),
            np.array([steps.shape[0]]),
        )
    )
    ax.plot(test_loss_xs, logger[test_loss_key], label="test loss")
    ax.legend()

    return fig, ax
