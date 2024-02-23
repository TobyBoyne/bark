from typing import Any, Union

import gpytorch as gpy
import torch
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise

from .forest import AlfalfaForest, AlfalfaTree


class AlfalfaTreeModelKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree_model: Union[AlfalfaTree, AlfalfaForest]):
        super().__init__()
        self.tree_model = tree_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return torch.as_tensor(self.tree_model.gram_matrix(x1, x2)).double()

    def get_extra_state(self):
        return {"tree_model": self.tree_model.as_dict()}

    def set_extra_state(self, state):
        d = state["tree_model"]
        if d["tree_model_type"] == "tree":
            self.tree_model = AlfalfaTree.from_dict(d)
        elif d["tree_model_type"] == "forest":
            self.tree_model = AlfalfaForest.from_dict(d)


class AlfalfaGP(gpy.models.ExactGP):
    def __init__(
        self, train_inputs, train_targets, likelihood, tree_model: AlfalfaTree
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()

        tree_kernel = AlfalfaTreeModelKernel(tree_model)
        self.covar_module = gpy.kernels.ScaleKernel(tree_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def tree_model(self) -> Union[AlfalfaTree, AlfalfaForest]:
        return self.covar_module.base_kernel.tree_model

    @classmethod
    def from_mcmc_samples(cls, model: "AlfalfaGP", samples):
        likelihood = gpy.likelihoods.GaussianLikelihood()
        all_trees = {"tree_model_type": "forest", "trees": []}
        for sample in samples:
            forest_dict = sample["covar_module.base_kernel._extra_state"]["tree_model"]
            all_trees["trees"] += forest_dict["trees"]

        tree_model = AlfalfaForest.from_dict(all_trees)
        tree_model.initialise(model.tree_model.space)
        gp = cls(model.train_inputs[0], model.train_targets, likelihood, tree_model)

        avg_noise = torch.mean(
            torch.tensor([s["likelihood.noise_covar.raw_noise"] for s in samples])
        )
        likelihood.noise = torch.nn.Softplus(avg_noise)

        avg_scale = torch.mean(
            torch.tensor([s["covar_module.raw_outputscale"] for s in samples])
        )
        gp.covar_module.outputscale = torch.nn.Softplus(avg_scale)

        return gp


class AlfalfaMOGP(gpy.models.ExactGP):
    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        tree_model: AlfalfaTree,
        num_tasks: int = 2,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()
        self.covar_module = AlfalfaTreeModelKernel(tree_model)
        self.task_covar_module = gpy.kernels.IndexKernel(num_tasks=num_tasks, rank=1)

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)

        covar = covar_x.mul(covar_i)
        return gpy.distributions.MultivariateNormal(mean_x, covar)

    @property
    def tree_model(self) -> Union[AlfalfaTree, AlfalfaForest]:
        return self.covar_module.tree_model


class MultitaskGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Likelihood for input-wise homo-skedastic noise, and task-wise hetero-skedastic, i.e. we learn a different (constant) noise level for each fidelity.

    [Folch 2023]

    Args:
        num_of_tasks : number of tasks in the multi output GP
        noise_prior : any prior you want to put on the noise
        noise_constraint : constraint to put on the noise
    """

    def __init__(
        self,
        num_tasks,
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        self.num_tasks = num_tasks
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> torch.Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # params contains training data
        task_idxs = params[0][-1]
        noise_base_covar_matrix = self.noise_covar(*params, shape=base_shape, **kwargs)
        # initialize masking
        mask = torch.zeros(size=noise_base_covar_matrix.shape)
        # for each task create a masking
        for task_num in range(self.num_tasks):
            # create vector of indexes
            task_idx_diag = (task_idxs == task_num).int().reshape(-1).diag()
            mask[..., task_num, :, :] = task_idx_diag
        # multiply covar by masking
        # there seems to be problems when base_shape is singleton, so we need to squeeze
        if base_shape == torch.Size([1]):
            noise_base_covar_matrix = noise_base_covar_matrix.squeeze(-1).mul(
                mask.squeeze(-1)
            )
            noise_covar_matrix = noise_base_covar_matrix.unsqueeze(-1).sum(dim=1)
        else:
            noise_covar_matrix = noise_base_covar_matrix.mul(mask).sum(dim=1)
        return noise_covar_matrix

    def forward(
        self,
        function_samples: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> base_distributions.Normal:
        noise = self._shaped_noise_covar(
            function_samples.shape, *params, **kwargs
        ).diag()
        return base_distributions.Normal(function_samples, noise.sqrt())

    def marginal(
        self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs).squeeze(0)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)
