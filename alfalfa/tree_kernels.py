import gpytorch as gpy
import torch
from beartype.typing import Any, Optional, Union
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise
from linear_operator.operators import DiagLinearOperator

from .forest import AlfalfaForest, AlfalfaTree
from .utils.space import Space


class AlfalfaTreeModelKernel(gpy.kernels.Kernel):
    is_stationary = False

    def __init__(self, tree_model: Optional[AlfalfaForest]):
        super().__init__()
        self.tree_model = tree_model

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, **params):
        if diag:
            return torch.ones(x1.shape[0])
        return torch.as_tensor(
            self.tree_model.gram_matrix(x1.detach().numpy(), x2.detach().numpy())
        )

    def get_extra_state(self):
        return {"tree_model": self.tree_model.as_dict()}

    def set_extra_state(self, state):
        d = state["tree_model"]
        self.tree_model = AlfalfaForest.from_dict(d)


class AlfalfaGP(gpy.models.ExactGP):
    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        tree_model: Optional[AlfalfaForest],
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


class AlfalfaMOGP(AlfalfaGP):
    def __init__(
        self,
        train_inputs,
        train_targets,
        likelihood,
        tree_model: Optional[AlfalfaForest],
        num_tasks: int = 2,
    ):
        super(AlfalfaGP, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpy.means.ZeroMean()
        self.covar_module = AlfalfaTreeModelKernel(tree_model)
        self.task_covar_module = gpy.kernels.IndexKernel(num_tasks=num_tasks, rank=1)
        self.num_tasks = num_tasks

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)

        covar = covar_x.mul(covar_i)
        return gpy.distributions.MultivariateNormal(mean_x, covar)

    @property
    def tree_model(self) -> Union[AlfalfaTree, AlfalfaForest]:
        return self.covar_module.tree_model


class AlfalfaMCMCModel:
    """A model generated from many MCMC samples of a GP"""

    def __init__(
        self,
        train_inputs,
        train_targets,
        samples,
        space: Space,
        sampling_seed: int,
        lag: int = 1,
    ):
        # samples contains a list of GP hyperparameters
        # need to take function samples
        self.samples = samples
        self.sampling_seed = sampling_seed

        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)

        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.space = space

        self.lag = lag

    def __call__(self, x, *args):
        function_samples = torch.zeros((len(self.samples), x.shape[0]))
        # fork rng allows us to draw consistent function samples
        with torch.random.fork_rng():
            torch.random.manual_seed(self.sampling_seed)
            for i, sample in enumerate(self.samples):
                if i % self.lag != 0:
                    continue

                gp = AlfalfaGP(
                    self.train_inputs,
                    self.train_targets,
                    gpy.likelihoods.GaussianLikelihood(),
                    None,
                )
                gp.load_state_dict(sample)
                gp.tree_model.initialise(self.space)
                gp.eval()

                output = gp.likelihood(gp(x))
                function_samples[i, :] = output.sample()

        mean = function_samples.mean(dim=0)
        var = DiagLinearOperator(function_samples.var(dim=0))
        return gpy.distributions.MultivariateNormal(mean, var)

    @property
    def likelihood(self):
        # likelihood returns the identity functions, so that this works nicely with plots
        return lambda x: x

    def state_dict(self):
        return self.samples

    def load_state_dict(self, state):
        self.samples = state


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

        all_tasks = torch.arange(self.num_tasks)[:, None]
        diag = torch.eq(all_tasks, task_idxs.mT)
        mask = DiagLinearOperator(diag)
        return (noise_base_covar_matrix @ mask).sum(dim=-3)

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


AnyModel = gpy.models.ExactGP | AlfalfaMCMCModel
