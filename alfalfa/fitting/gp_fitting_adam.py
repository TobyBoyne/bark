"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""

import gpytorch as gpy
import torch


def fit_gp_adam(model: gpy.models.ExactGP, verbose: bool = False) -> None:
    """Fit a Gaussian process model using the Adam optimiser.

    Fit a GP where all parameters to be optimised are PyTorch parameters,
    allowing for differentiable optimisation.

    This function supports multi-fidelity, where `model.train_inputs[1]` contains
    the task index.

    Args:
        model (gpy.models.ExactGP): the GP model
    """
    (x, *args) = model.train_inputs
    y = model.train_targets
    likelihood = model.likelihood

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter := 500):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x, *args)
        # Calc loss and backprop gradients
        loss = -mll(output, y, args) if args else -mll(output, y)
        loss.backward()
        if (i + 1) % 100 == 0 and verbose:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
        optimizer.step()
