import gpytorch as gpy
import torch
from beartype.typing import Optional, Union
from torch.distributions import Categorical, MixtureSameFamily

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


class AlfalfaMixtureModel:
    """A model generated from many MCMC samples of a GP.

    The posterior is a mixture of Gaussians."""

    def __init__(
        self,
        train_inputs,
        train_targets,
        samples,
        space: Space,
        sampling_seed: int = None,
    ):
        # samples contains a list of GP hyperparameters
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

        for i, gp in enumerate(self.gp_samples_iter()):
            output = gp(x)
            if predict_y:
                output = gp.likelihood(output)

            f_mean[i, :] = output.loc
            f_var[i, :, :] = output.covariance_matrix

        # add jitter
        f_var += 1e-6 * torch.eye(f_var.shape[-1])

        mix = Categorical(probs=torch.ones((len(self.samples),)))
        comp = torch.distributions.MultivariateNormal(f_mean, f_var)
        return MixtureSameFamily(mix, comp)

    def gp_samples_iter(self, likelihood=None):
        """Return an iterator over the GP samples.

        Note that the GP is the same underlying object, so saving a reference
        to the GP will result in identical samples."""
        if likelihood is not None:
            raise NotImplementedError()

        gp = AlfalfaGP(
            self.train_inputs,
            self.train_targets,
            gpy.likelihoods.GaussianLikelihood(),
            None,
        )
        gp.eval()
        for sample in self.samples:
            gp.load_state_dict(sample)
            gp.tree_model.initialise(self.space)
            yield gp

    @property
    def likelihood(self):
        # likelihood returns the identity functions, so that this works nicely with plots
        return lambda x: x

    def state_dict(self):
        return self.samples

    def load_state_dict(self, state):
        self.samples = state
