import gpytorch as gpy
import torch
from beartype.typing import Optional, Union
from gpytorch.distributions import MultivariateNormal

from ..forest import AlfalfaForest, AlfalfaTree
from ..utils.space import Space
from .tree_model_kernel import AlfalfaTreeModelKernel


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


class AlfalfaSampledModel:
    """A model generated from many MCMC samples of a GP"""

    def __init__(
        self,
        train_inputs,
        train_targets,
        samples,
        space: Space,
        sampling_seed: int,
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

    def __call__(self, x, *args, predict_y=False):
        f_mean = torch.zeros((len(self.samples), x.shape[0]))
        f_var = torch.zeros((len(self.samples), x.shape[0], x.shape[0]))
        gp = AlfalfaGP(
            self.train_inputs,
            self.train_targets,
            gpy.likelihoods.GaussianLikelihood(),
            None,
        )

        for i, sample in enumerate(self.samples):
            gp.load_state_dict(sample)
            gp.tree_model.initialise(self.space)
            gp.eval()

            output = gp(x)
            if predict_y:
                output = gp.likelihood(output)

            f_mean[i, :] = output.loc
            f_var[i, :, :] = output.covariance_matrix

        output_mean = torch.mean(f_mean, dim=0)
        output_var = torch.mean(f_var, dim=0)
        return MultivariateNormal(output_mean, output_var)

    @property
    def likelihood(self):
        # likelihood returns the identity functions, so that this works nicely with plots
        return lambda x: x

    def state_dict(self):
        return self.samples

    def load_state_dict(self, state):
        self.samples = state
