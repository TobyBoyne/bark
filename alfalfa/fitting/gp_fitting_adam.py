"""Convert an LGBM tree to an instance of Alternating Tree for comparison"""

import gpytorch as gpy
import torch


def fit_gp_adam(model: gpy.models.ExactGP):
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
    mll = KwdExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter := 100):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x, *args)
        # Calc loss and backprop gradients
        (task_idxs,) = args
        loss = -mll(output, y, task_idxs=task_idxs)
        loss.backward()
        if (i + 1) % 100 == 0:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
        optimizer.step()


class KwdExactMarginalLogLikelihood(gpy.mlls.ExactMarginalLogLikelihood):
    def forward(self, function_dist, target, *params, **kwargs):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, gpy.distributions.MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihood can only operate on Gaussian random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params, **kwargs)
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)
